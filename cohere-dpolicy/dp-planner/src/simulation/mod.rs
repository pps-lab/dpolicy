//! This module most importantly offers the method [run_simulation] which contains high-level
//! management code which is the same for all types of allocation, as well as some helper methods.
//! Methods from the [logging module](crate::logging) are also called here to produce the
//! various logs.

pub mod util;

use log::{info, trace};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::ops::{Add, AddAssign, Sub};
use std::sync::{Arc, RwLock};
use std::time::Instant;

use crate::allocation::AllocationStatus;
use crate::block::BlockId;
use crate::config::OutputPaths;
use crate::logging::{RuntimeKind, RuntimeMeasurement};
use crate::{
    allocation::{AllocationRound, ResourceAllocation},
    block::Block,
    logging,
    request::{Request, RequestId},
    schema::Schema, Cli,
};
use serde::{Deserialize, Serialize};
use crate::composition::block_composition_pa::NArrayCache;

/// Contains various information needed to run the simulation. See the documentation about the
/// individual fields for more information.
pub struct SimulationConfig {
    /// The batching strategy define whether we will batch with a fixed batch size or batch based on the created field present on each request.
    pub(crate) batching_strategy: BatchingStrategy,
    /// Defines in how many rounds a request is considered for allocation at most. Note that if this
    /// number is greater than 1, more than batch_size many requests may be present in a single
    /// round, potentially slowing down allocation considerably. If a request still has "rounds left"
    /// after the whole simulation, it is counted as a request that is rejected, but is still available
    /// for allocation later. The information how many times it was already considered is lost in
    /// this case.
    pub(crate) timeout_rounds: usize,
    /// The round in which the simulation starts. This is important if the [budget](Budget) is an
    /// [unlocking budget](Budget::UnlockingBudget), as how much budget is unlocked depends on the
    /// current round and on when a block was created.
    pub(crate) start_round: RoundId,
    /// The paths for the various outputs during/at the end of the simulation. See [OutputPaths] for
    /// more details.
    pub(crate) output_paths: OutputPaths,
    /// Whether we want to include request rejections that were not final (only matters if
    /// [keep rejected requests](struct.SimulationConfig.html#structfield.keep_rejected_requests)
    /// is enabled)
    pub(crate) log_nonfinal_rejections: bool,
}

/// Also contains information to run the simulation like [SimulationConfig], but in contrast
/// this information might be changed during the simulation due to global alpha reduction, while
/// [SimulationConfig] is not mutable.
#[derive(Clone)]
pub struct ConfigAndSchema {
    /// The schema matching the passed requests
    pub(crate) schema: Schema,
    /// The input given via the command line, possibly modified by
    /// [global alpha reduction](crate::global_reduce_alphas).
    pub(crate) config: Cli,
}

/// Contains various datastructures containing requests, which are necessary to run the simulation
pub struct RequestCollection {
    pub(crate) sorted_candidates: Vec<Request>,
    pub(crate) request_history: HashMap<RequestId, Request>,
    pub(crate) rejected_requests: BTreeMap<RequestId, Request>,
    pub(crate) accepted: BTreeMap<RequestId, HashSet<BlockId>>,
    pub(crate) remaining_requests: BTreeMap<RequestId, Request>,
}

#[derive(Deserialize, Debug, Clone, Copy, Hash, Serialize)]
pub enum BatchingStrategy {
    ByRequestCreated,
    ByBatchSize(usize),
}

#[derive(Deserialize, Debug, Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub struct RoundId(pub usize);

impl AddAssign<usize> for RoundId {
    fn add_assign(&mut self, rhs: usize) {
        self.0 += rhs
    }
}

impl Add for RoundId {
    type Output = RoundId;

    fn add(self, rhs: Self) -> Self::Output {
        RoundId(self.0 + rhs.0)
    }
}

impl Sub for RoundId {
    type Output = RoundId;

    fn sub(self, rhs: Self) -> Self::Output {
        RoundId(self.0 - rhs.0)
    }
}


impl RoundId{
    pub fn to_usize(&self) -> usize {
        self.0
    }
}
pub fn run_simulation(
    request_collection: &mut RequestCollection,
    blocks: &mut HashMap<BlockId, Block>,
    allocator: &mut AllocationRound,
    simulation_config: &SimulationConfig,
    config_and_schema: &mut ConfigAndSchema,
) {
    assert!(
        simulation_config.timeout_rounds > 0,
        "timeout_rounds must be > 0"
    );
    // initialize logger
    let mut req_logger =
        csv::Writer::from_path(&simulation_config.output_paths.req_log_output_path)
            .expect("Couldn't open request logger output file");
    let mut round_logger =
        csv::Writer::from_path(&simulation_config.output_paths.round_log_output_path)
            .expect("Couldn't open round logger output file");

    let mut runtime_measurements: Vec<RuntimeMeasurement> = Vec::new();
    let mut runtime_logger =
        csv::Writer::from_path(&simulation_config.output_paths.runtime_log_output_path)
            .expect("Couldn't open runtime logger output file");

    // Clear applied_request cache
    let cache_requests = std::env::var("CACHE_REQUEST_COST").unwrap_or("0".to_string());
    if cache_requests == "1" {
        println!("Clearing request cost cache");
        let tmp_dir = std::env::temp_dir();
        if tmp_dir.exists() {
            for block in blocks.keys() {
                let tmp_file_path = tmp_dir.join(format!("cum_request_cost_{block}.bin"));
                if tmp_file_path.exists() {
                    println!("Removing file: {:?}", tmp_file_path);
                    std::fs::remove_file(tmp_file_path).unwrap();
                }
            }
        }
    }

    // The current round in the simulation. Note that for online allocation methods (like greedy),
    // one simulation round might correspond to multiple allocation rounds.
    let mut simulation_round = simulation_config.start_round;
    let mut request_start_rounds: HashMap<RequestId, RoundId> = HashMap::new();

    // for each round > start_round, contains block ids of blocks that joined this round
    // while all blocks that are currently part of the system are under the current_round
    let mut blocks_per_round: BTreeMap<RoundId, Vec<(BlockId, Block)>> = BTreeMap::new();

    let block_ids_to_remove: Vec<BlockId> = blocks
        .iter()
        .filter(|(_block_id, block)| block.created >= simulation_round)
        .map(|(block_id, _block)| *block_id)
        .collect();

    for block_id in block_ids_to_remove {
        let block = blocks.remove(&block_id).unwrap();
        blocks_per_round
            .entry(block.created)
            .or_default()
            .push((block_id, block));
    }
    // blocks: contains all blocks with a created field < simulation_round
    // blocks_per_round contains all blocks with a created field >= simulation_round (grouped by round)

    let orig_n_candidates = request_collection.sorted_candidates.len();
    let orig_history_size = request_collection.request_history.len();

    let mut curr_candidate_requests: BTreeMap<RequestId, Request> = BTreeMap::new();

    let mut round_instant: Option<Instant> = None;
    let start_instant = Instant::now();
    let rounds_instant: Instant = Instant::now();

    let mut total_profit = 0u64;

    // TODO: Delete block cache when done with block
    let block_narray_cache: &mut HashMap<BlockId, Arc<RwLock<NArrayCache>>> = &mut HashMap::new();

    loop {
        let mut round_total_meas = RuntimeMeasurement::start(RuntimeKind::TotalRound);
        let mut round_setup_meas = RuntimeMeasurement::start(RuntimeKind::RoundSetup);

        assert!(
            runtime_measurements.is_empty(),
            "runtime_measurements should be empty at the beginning of each round"
        );

        util::log_remaining_requests(
            &*request_collection,
            simulation_config,
            orig_n_candidates,
            &mut round_instant,
            &start_instant,
        );

        // Check if we have already processed all request - in which case we can stop the simulation.
        if request_collection.sorted_candidates.is_empty() {
            break;
        }

        // move the new batch of requests from request.collection.sorted_candidates to curr_candidate_requests
        let (newly_available_requests, is_final_round) = util::pre_round_request_batch_update(
            simulation_round,
            simulation_config.batching_strategy,
            request_collection,
            &mut curr_candidate_requests,
            &mut request_start_rounds,
        );

        // activate and retire blocks so that blocks only contain blocks that are active in the current round
        util::pre_round_blocks_update(simulation_round, blocks, &mut blocks_per_round);

        // update the budget of the active blocks in `blocks` (budget unlocking)
        util::update_block_unlocked_budget(
            blocks,
            config_and_schema.config.allocation().budget_config(),
            simulation_round,
        );


        trace!(
            "Current candidate requests: {:?}",
            &curr_candidate_requests.keys()
        );

        //let expected_num_active_blocks = config_and_schema.schema.block_sliding_window_size * config_and_schema.schema.privacy_units.len();
        //assert!(blocks.len() == expected_num_active_blocks, "Round: {:?}   Expected number of active blocks: {:?}, actual number of active blocks: {:?}", simulation_round, expected_num_active_blocks, blocks.len());

        info!("Starting allocation in round={:?} n_blocks={:?} n_candidate_requests={:?}   n_request_allocated_until_now={:?}", simulation_round, blocks.len(), curr_candidate_requests.len(), request_collection.request_history.len());

        runtime_measurements.push(round_setup_meas.stop());

        let mut round_allocation_meas = RuntimeMeasurement::start(RuntimeKind::RoundAllocation);

        // Run a round of allocation
        let (assignment, allocation_status): (ResourceAllocation, AllocationStatus) = allocator
            .round(
                &request_collection.request_history,
                blocks,
                &mut curr_candidate_requests,
                &config_and_schema.schema,
                &mut runtime_measurements,
                block_narray_cache
            );
        runtime_measurements.push(round_allocation_meas.stop());
        runtime_measurements.push(round_total_meas.stop());

        util::process_round_results(
            request_collection,
            blocks,
            simulation_config,
            &mut req_logger,
            &mut curr_candidate_requests,
            simulation_round,
            &mut total_profit,
            is_final_round,
            &assignment,
            &*config_and_schema,
            &request_start_rounds,
        );

        logging::write_round_log_row(
            &mut round_logger,
            simulation_round,
            simulation_config,
            config_and_schema,
            newly_available_requests,
            BTreeSet::from_iter(assignment.accepted.keys().copied()),
            allocation_status,
            config_and_schema.config.allocation(),
            &request_collection.request_history,
            &*blocks,
        );

        logging::write_runtime_log(
            &mut runtime_logger,
            simulation_round,
            &mut runtime_measurements,
        );

        simulation_round += 1;
    }

    // Add all blocks again to blocks (necessary if there were no candidate requests, or not enough
    // rounds to add all blocks)
    for (_, block) in blocks_per_round.into_values().flatten() {
        let inserted = blocks.insert(block.id, block);
        assert!(inserted.is_none());
    }

    util::process_simulation_results(
        request_collection,
        &*blocks,
        simulation_config,
        &*config_and_schema,
        orig_n_candidates,
        orig_history_size,
        curr_candidate_requests,
        total_profit,
    );

    info!(
        "Total Time to run simulation: {} seconds",
        rounds_instant.elapsed().as_millis() as f64 / 1000f64,
    );
}

#[cfg(test)]
mod tests {

}

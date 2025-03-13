//! This crate offers a variety of methods assigning requests to blocks, such that the
//! resulting allocation fulfills a certain differential privacy constraint.

use crate::block::{Block, BlockId};
use crate::config::Cli;
use clap::Parser;
use itertools::Itertools;
use log::{info, trace};
use std::collections::{BTreeMap, HashMap, HashSet};

use crate::dprivacy::budget::OptimalBudget;
use crate::dprivacy::{
    Accounting, AccountingType,
};
use crate::request::{Request, RequestId};
use crate::schema::Schema;
use crate::simulation::{
    BatchingStrategy, ConfigAndSchema, RequestCollection, RoundId, SimulationConfig,
};

pub mod allocation;
pub mod block;
pub mod composition;
pub mod config;
pub mod dprivacy;
pub mod logging;
pub mod request;
pub mod schema;
pub mod simulation;
pub mod util;

pub fn run_program() {
    #[cfg(debug_assertions)]
    info!("Debug mode enabled");

    let config: Cli = config::Cli::parse();

    trace!("Input config: {:?}", config);

    // Check that the output paths are valid
    let output_paths = config::check_output_paths(&config);

    // check that there are no misconfigurations
    config.check_config();


    // load and init schema
    let schema = schema::load_schema(config.input.schema.clone(), config.total_budget(), config.block_sliding_window_size())
        .expect("loading schema failed");

    trace!("Loading candidate requests...");
    // loads candidate requests and converts them to internal format
    let candidate_requests = request::load_requests(
        config.input.requests.clone(),
        &schema,
    )
    .expect("loading requests failed");
    trace!("Loaded {} candidate requests", candidate_requests.len());



    trace!("Loading history requests...");
    let request_history: HashMap<RequestId, Request> = match &config.input.history {
        Some(path) => request::load_requests(
            path.clone(),
            &schema,
        )
        .expect("loading history failed"),
        None => HashMap::new(),
    };

    trace!("Loaded {} history requests", request_history.len());

    // TODO: Check that history is feasible, and does not violate any budget constraint

    // make sure request ids are unique also between candidate_requests and request_history
    for (rid, _request) in candidate_requests.iter() {
        assert!(
            !request_history.contains_key(rid),
            "A request id was used for candidate requests and the request history"
        )
    }

    trace!("Loading blocks...");
    let mut blocks: HashMap<BlockId, Block> = block::load_blocks(
        config.input.blocks.clone(),
        &request_history,
        &schema,
        &config.total_budget(),
    )
    .expect("loading of blocks failed");
    trace!("Loaded {} blocks", blocks.len());


    let rejected_requests: BTreeMap<RequestId, Request> = BTreeMap::new();
    // request ids that were accepted during the run of this program
    // invariant: every request id in accepted in a key in history_requests
    let accepted: BTreeMap<RequestId, HashSet<BlockId>> = BTreeMap::new();

    let mut sorted_candidates: Vec<Request> = candidate_requests
        .into_iter()
        .map(|(_, request)| request)
        .sorted_by(|r1, r2| Ord::cmp(&r1.request_id, &r2.request_id))
        .collect();

    let is_sorted_by_created = sorted_candidates
        .windows(2)
        .all(|w| w[0].created <= w[1].created);
    assert!(
        is_sorted_by_created,
        "Sorting requests by request id leads to requests that are not sorted by the created field"
    );

    let mut n_candidates = sorted_candidates.len();

    let mut allocator = allocation::construct_allocator(config.allocation());

    let simulation_config: SimulationConfig;
    let mut config_and_schema: ConfigAndSchema;

    let mut request_collection: RequestCollection;

    let config_clone = config.clone();

    match config.mode {
        config::Mode::Simulate {
            allocation: _allocation,
            timeout_rounds,
            max_requests,
        } => {
            let mut remaining_requests: BTreeMap<RequestId, Request> = BTreeMap::new();
            if let Some(max_req) = max_requests {
                assert!(
                    max_req <= n_candidates,
                    "max_requests needs to be <= the number of candidate requests"
                );
                remaining_requests.extend(
                    sorted_candidates
                        .drain(max_req..n_candidates)
                        .map(|req| (req.request_id, req)),
                );
                n_candidates = max_req
            }

            let batching_strategy = BatchingStrategy::ByRequestCreated;

            simulation_config = SimulationConfig {
                batching_strategy,
                timeout_rounds,
                start_round: sorted_candidates[0].created,
                output_paths,
                log_nonfinal_rejections: config.output_config.log_nonfinal_rejections,
            };

            config_and_schema = ConfigAndSchema {
                schema,
                config: config_clone,
            };

            request_collection = RequestCollection {
                sorted_candidates,
                request_history,
                rejected_requests,
                accepted,
                remaining_requests,
            };

            simulation::run_simulation(
                &mut request_collection,
                &mut blocks,
                &mut allocator,
                &simulation_config,
                &mut config_and_schema,
            )
        }

        config::Mode::Round {
            allocation: _allocation,
            i,
            ..
        } => {
            for request in sorted_candidates.iter() {
                assert!(
                    request.created <= RoundId(i),
                    "Tried to run round {}, but request with id {} is created only at time {:?}",
                    i,
                    request.request_id.0,
                    request.created
                );
            }

            simulation_config = SimulationConfig {
                batching_strategy: BatchingStrategy::ByBatchSize(sorted_candidates.len()), // If we are in round mode, we form a batch by taking all the candidate requests
                timeout_rounds: 1,
                start_round: RoundId(i),
                output_paths,
                log_nonfinal_rejections: config.output_config.log_nonfinal_rejections,
            };

            config_and_schema = ConfigAndSchema {
                schema,
                config: config_clone,
            };

            request_collection = RequestCollection {
                sorted_candidates,
                request_history,
                rejected_requests,
                accepted,
                remaining_requests: BTreeMap::new(),
            };

            simulation::run_simulation(
                &mut request_collection,
                &mut blocks,
                &mut allocator,
                &simulation_config,
                &mut config_and_schema,
            );
        }
    }

    assert_eq!(
        request_collection.accepted.len() + request_collection.rejected_requests.len(),
        n_candidates,
        "Lost some requests while allocating"
    );

    info!(
        "Accepted {} requests, rejected {} requests",
        request_collection.accepted.len(),
        request_collection.rejected_requests.len()
    );
}

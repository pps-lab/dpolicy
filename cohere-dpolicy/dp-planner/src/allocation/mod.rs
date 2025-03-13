
pub mod efficiency_based;
pub mod utils;

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::{Arc, RwLock};
use crate::allocation::efficiency_based::EfficiencyBased;
use crate::allocation::BlockCompWrapper::{
    BlockCompositionPartAttributesVariant, BlockCompositionVariant,
};
use crate::composition::block_composition::{build_block_composition, BlockComposition};
use crate::composition::block_composition_pa::{build_block_part_attributes, BlockCompositionPartAttributes, NArrayCache};
use crate::composition::{CompositionConstraint, ProblemFormulation};
use crate::config::BudgetType;
use crate::dprivacy::budget::{RdpMinBudget, SegmentBudget};
use crate::logging::{DpfStats, RuntimeKind, RuntimeMeasurement};
use crate::{
    block::{Block, BlockId},
    config::{AllocationConfig, CompositionConfig},
    request::{Request, RequestId},
    schema::Schema,
};
use crate::OptimalBudget;

/// Contains a final allocation after an allocation method was run.
/// For accepted requests, contains the request id and block ids of allocated blocks,
/// and for rejected requests, just the request ids.
#[derive(Debug, Clone)]
pub struct ResourceAllocation {
    pub accepted: HashMap<RequestId, HashSet<BlockId>>,

    pub rejected: HashSet<RequestId>,
}

/// Implements composition constraint, by abstracting over the underlying BlockComposition
/// (either with or without PA, depending on initialization)
pub enum BlockCompWrapper {
    BlockCompositionPartAttributesVariant(BlockCompositionPartAttributes),
    BlockCompositionVariant(BlockComposition),
}

/// Exposes a [AllocationRound::round()] method, which provides a unified interface to run a round of Dpf,
/// Ilp or Greedy allocation, while storing data valid across multiple rounds.
/// Constructed via [construct_allocator]
#[allow(clippy::large_enum_variant)]
pub enum AllocationRound {
//    Dpf(Dpf, BlockCompWrapper, BudgetType),
//    Ilp(Ilp, BlockCompWrapper, BudgetType),
//    Greedy(Greedy, BlockCompWrapper, BudgetType),
    EfficiencyBased(EfficiencyBased, BlockCompWrapper, BudgetType),
}

/// Used for passing the status of the ilp model after optimization to simulation.rs - if the
/// allocation method is not ilp, it is just None.
pub enum AllocationStatus {
    //GreedyStatus(GreedyStats),
    DpfStatus(DpfStats),
    //IlpRegularStatus(grb::Status, IlpStats),
    //IlpOtherStatus(&'static str, IlpStats),
}

impl AllocationStatus {
    /// Get the number of contested segments when the problem formulation is first constructed.
    pub fn get_initial_num_contested_segments(&self) -> Option<usize> {
        match self {
            AllocationStatus::DpfStatus(dpf_stats) => {
                Some(dpf_stats.num_contested_segments_initially)
            }
        }
    }

    pub fn to_string(&self) -> &'static str {
        match self {
            AllocationStatus::DpfStatus(..) => "n/a",
        }
    }


}

impl CompositionConstraint for BlockCompWrapper {
    fn build_problem_formulation<M: SegmentBudget>(
        &self,
        blocks: &HashMap<BlockId, Block>,
        candidate_requests: &BTreeMap<RequestId, Request>,
        history_requests: &HashMap<RequestId, Request>,
        schema: &Schema,
        runtime_measurements: &mut Vec<RuntimeMeasurement>,
        block_narray_cache: &mut HashMap<BlockId, Arc<RwLock<NArrayCache>>>
    ) -> ProblemFormulation<M> {
        let mut prob_meas = RuntimeMeasurement::start(RuntimeKind::BuildProblemFormulation);
        let pf = match self {
            BlockCompositionPartAttributesVariant(bc) => bc.build_problem_formulation::<M>(
                blocks,
                candidate_requests,
                history_requests,
                schema,
                runtime_measurements,
                block_narray_cache
            ),
            BlockCompositionVariant(bc) => bc.build_problem_formulation(
                blocks,
                candidate_requests,
                history_requests,
                schema,
                runtime_measurements,
                block_narray_cache
            ),
        };
        runtime_measurements.push(prob_meas.stop());
        pf
    }
}

/// Constructs an allocator of type [AllocationRound] to later run rounds of allocation on
/// # Arguments
/// * `allocation_config` - defines whether or not partitioning attributes are used, and the type
/// of allocation algorithm (Dpf, Greedy, or Ilp)
/// * `sorted_candidate_requests` - all requests that may later appear. Needed to fix for each
/// request which blocks are selected in Dpf, but not used for allocation decisions
/// * `blocks` - all blocks available at any point during the program. Also used to fix for each
/// request which blocks are selected in Dpf, but not used for allocation decisions
// TODO: Remove sorted candidate requests from here, and make new decisions each round (with caching for consistency)
pub fn construct_allocator(allocation_config: &AllocationConfig) -> AllocationRound {
    // TODO could shorten code by using macros or implementing trait for batch schedulers
    match allocation_config {
        AllocationConfig::EfficiencyBased {
            algo_type,
            block_selector_seed,
        } => {
            let composition = algo_type.get_composition();
            let (_, block_comp_wrapper, budget_type) = match composition {
                CompositionConfig::BlockCompositionPa {
                    budget,
                    algo,
                    budget_type,
                    num_threads,
                } => (
                    budget,
                    BlockCompositionPartAttributesVariant(build_block_part_attributes(*algo, num_threads.clone())),
                    budget_type,
                ),
                CompositionConfig::BlockComposition {
                    budget,
                    budget_type,
                } => (
                    budget,
                    BlockCompositionVariant(build_block_composition()),
                    budget_type,
                ),
            };

            let allocator = efficiency_based::EfficiencyBased::construct_allocator(
                *block_selector_seed,
                algo_type.clone(),
            );

            AllocationRound::EfficiencyBased(allocator, block_comp_wrapper, budget_type.clone())
        }
    }
}

impl AllocationRound {
    /// Provides a unified interface to run a round of the chosen allocation algorithm and the parameters
    /// set in [construct_allocator]
    /// # Arguments
    /// * `request_history` - all past accepted requests (initial history + accepted during simulation)
    /// * `blocks` - all blocks that are available in the current round
    /// * `request_batch` - requests to be decided on this round (either they are accepted or rejected)
    /// * `schema` - the common schema of all requests

    pub fn round(
        &mut self,
        request_history: &HashMap<RequestId, Request>,
        available_blocks: &HashMap<BlockId, Block>,
        candidate_requests: &mut BTreeMap<RequestId, Request>,
        schema: &Schema,
        runtime_measurements: &mut Vec<RuntimeMeasurement>,
        block_narray_cache: &mut HashMap<BlockId, Arc<RwLock<NArrayCache>>>,
    ) -> (ResourceAllocation, AllocationStatus) {
        match self {
            AllocationRound::EfficiencyBased(allocator, block_comp_wrapper, budget_type) => {
                match budget_type {
                    BudgetType::OptimalBudget => allocator.round::<OptimalBudget>(
                        candidate_requests,
                        request_history,
                        available_blocks,
                        schema,
                        block_comp_wrapper,
                        runtime_measurements,
                        block_narray_cache
                    ),
                    BudgetType::RdpMinBudget => allocator.round::<RdpMinBudget>(
                        candidate_requests,
                        request_history,
                        available_blocks,
                        schema,
                        block_comp_wrapper,
                        runtime_measurements,
                        block_narray_cache
                    ),
                }
            }
        }
    }
}

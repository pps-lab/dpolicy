use itertools::Itertools;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::{Arc, RwLock};
use crate::composition::ProblemFormulation;
use crate::dprivacy::privacy_unit::is_selected_block;
use crate::logging::RuntimeMeasurement;
use crate::{
    block::{Block, BlockId},
    composition::BlockSegment,
    dprivacy::{budget::SegmentBudget, AccountingType},
    request::{Request, RequestId},
    schema::Schema,
};
use crate::composition::block_composition_pa::NArrayCache;
use super::{BlockConstraints, CompositionConstraint};

pub struct BlockComposition {}

pub fn build_block_composition() -> BlockComposition {
    BlockComposition {}
}

impl CompositionConstraint for BlockComposition {
    fn build_problem_formulation<M: SegmentBudget>(
        &self,
        blocks: &HashMap<BlockId, Block>,
        candidate_requests: &BTreeMap<RequestId, Request>,
        history_requests: &HashMap<RequestId, Request>,
        _schema: &Schema,
        _runtime_measurements: &mut Vec<RuntimeMeasurement>,
        _block_narray_cache: &mut HashMap<BlockId, Arc<RwLock<NArrayCache>>>
    ) -> ProblemFormulation<M> {


        for (_, b) in blocks.iter(){
            for (_, r) in candidate_requests.iter(){
                assert!(is_selected_block(r, b), "At the moment for block compoisiton without PA, we only correctly support the case where all requests, select all blocks.");
            }
        }


        let block_constraints: HashMap<BlockId, BlockConstraints<M>> = blocks
            .iter()
            .map(|(block_id, block)| {
                (
                    *block_id,
                    build_block_constraint(
                        block,
                        history_requests,
                        candidate_requests,
                        block.default_unlocked_budget.clone().expect("For block_composition (without pa), the unlocked budget needs to be set"),
                    ),
                )
            })
            .collect();

        ProblemFormulation::new(block_constraints, candidate_requests, blocks)
    }
}

fn build_block_constraint<M: SegmentBudget>(
    block: &Block,
    history: &HashMap<RequestId, Request>,
    candidate_requests: &BTreeMap<RequestId, Request>,
    budget: AccountingType,
) -> BlockConstraints<M> {
    let block_history_cost = block
        .request_history
        .iter()
        .map(|request_id| {
            history
                .get(request_id)
                .expect("missing request")
                .request_cost(&block.privacy_unit)
                .clone()
        })
        .reduce(|agg, item| agg + item);

    let remaining_block_budget = match block_history_cost {
        Some(block_history_cost) => budget - block_history_cost, // subtract cost of history from budget
        None => budget, // no history -> full budget available
    };

    let mut remaining_budget = M::new();
    remaining_budget.add_budget_constraint(&remaining_block_budget);

    let (candidates, rejected): (Vec<&Request>, Vec<&Request>) = candidate_requests
        .values()
        .partition(|request| remaining_budget.is_budget_sufficient(request.request_cost(&block.privacy_unit)));



    let request_cost_sum: Option<AccountingType> = candidates
        .iter()
        .map(|request| request.request_cost(&block.privacy_unit))
        .fold(None, |agg, other|{
            match agg {
                        None => Some(other.clone()),
                        Some(c_acc) => Some(c_acc + other),
                    }});

    let candidate_request_ids: HashSet<RequestId> =
        candidates.iter().map(|r| r.request_id).collect();

    let (acceptable, contested, contested_segments) = match request_cost_sum {
        Some(request_cost_sum) if remaining_budget.is_budget_sufficient(&request_cost_sum) => {
            (candidate_request_ids, HashSet::new(), Vec::new())
        } // sufficient budget -> all accepted
        Some(_) => {
            let request_ids = Some(candidate_request_ids.iter().copied().sorted().collect());
            (
                HashSet::new(),
                candidate_request_ids,
                vec![BlockSegment {
                    id: 0,
                    request_ids,
                    remaining_budget,
                }],
            )
        } // not sufficient budget -> all contested
        None => (HashSet::new(), HashSet::new(), Vec::new()), // no candidates left -> no accepted no contested (all rejected)
    };

    BlockConstraints {
        acceptable,
        contested,
        rejected: rejected.iter().map(|r| r.request_id).collect(),
        contested_segments,
    }
}

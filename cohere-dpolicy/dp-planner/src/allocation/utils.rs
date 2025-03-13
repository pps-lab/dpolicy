//! Utility functions for allocation.

use crate::allocation::ResourceAllocation;
use crate::block::BlockId;
use crate::composition::ProblemFormulation;
use crate::dprivacy::budget::SegmentBudget;
use crate::request::{Request, RequestId};
use std::collections::{BTreeMap, HashSet};

/// Allocates a request if selected blocks can be allocated, else does not allocate.
/// In any case, resource_allocation is updated.
pub fn try_allocation<M: SegmentBudget>(
    candidate_requests: &BTreeMap<RequestId, Request>,
    pf: &mut ProblemFormulation<M>,
    resource_allocation: &mut ResourceAllocation,
    rid: RequestId,
    selected_blocks: HashSet<BlockId>,
    acc_bids: Vec<BlockId>,
    con_bids: Vec<BlockId>,
) {
    let allocatable_blocks = {
        let mut allocatable_blocks: HashSet<BlockId> = HashSet::from_iter(acc_bids);
        allocatable_blocks.extend(con_bids);
        allocatable_blocks
    };
    // println!("{}", pf.visualize_string());
    if selected_blocks
        .iter()
        .all(|bid| allocatable_blocks.contains(bid))
    {
        assert!(
            selected_blocks.len() >=
            candidate_requests[&rid].num_blocks.unwrap(),
            "Not enough blocks to accept"
        );
        pf.allocate_request(rid, &selected_blocks, candidate_requests)
            .expect("Allocating request failed");
        resource_allocation.accepted.insert(rid, selected_blocks);
    } else {
        resource_allocation.rejected.insert(rid);
    }
}

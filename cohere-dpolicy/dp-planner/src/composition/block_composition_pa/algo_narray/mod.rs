use crate::block::{Block, BlockId};
use crate::composition::{BlockConstraints, BlockSegment};
use crate::dprivacy::budget::SegmentBudget;
use crate::dprivacy::privacy_unit::{is_selected_block, PrivacyUnit};
use crate::dprivacy::rdp_alphas_accounting::RdpAlphas;
use crate::request::{Conjunction, Request, RequestId};
use crate::schema::Schema;

use super::{NArrayCache, Segmentation};

use float_cmp::{ApproxEq, F64Margin};
use itertools::Itertools;

use fasthash::{sea::Hash64, FastHash};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use std::collections::{HashMap, HashSet};
use std::{fs};
use std::fs::File;
use bincode::{config, Decode, Encode};
use serde::{Deserialize, Serialize};

pub(crate) mod narray;

use crate::dprivacy::{Accounting, AccountingType, AdpAccounting};
use narray::{Dimension, Index, NArray};
use crate::logging::{RuntimeKind, RuntimeMeasurement};

// TODO [nku] [later] write tests
pub struct NArraySegmentation<'r, 's> {
    dimension: Dimension,
    requested_budget: NArray<VirtualBlockRequested>,
    request_batch: Vec<&'r Request>,
    schema: &'s Schema,
}

#[derive(Clone, Serialize, Deserialize, Encode, Decode)]
pub struct VirtualBlockBudget {
    budget: Option<AccountingType>, // TODO: Using AccountingType here is not the best idea because the size of each element depends on the largest enum variant (which is rdp16)
    prev_request_id: Option<RequestId>,
}

#[derive(Clone)]
struct VirtualBlockRequested {
    request_hash: u64,
    request_count: u32,
    //cost: AccountingType,
    prev_request_id: Option<RequestId>,
}

impl<'r, 's> Segmentation<'r, 's> for NArraySegmentation<'r, 's> {
    fn new(request_batch: Vec<&'r Request>, schema: &'s Schema) -> Self {
        let dimension: Vec<usize> = schema.attributes.iter().map(|attr| attr.len()).collect();
        let dimension = Dimension::new(&dimension);


        // calculate requested budget
        let block = VirtualBlockRequested::new(schema);
        let mut requested_budget = narray::build(&dimension, block);
        calculate_requested_budget(&request_batch, &mut requested_budget, schema);

        println!("  requested_budget.size()={:?}", requested_budget.size());

        NArraySegmentation {
            dimension,
            requested_budget,
            request_batch,
            schema,
        }
    }

    fn new_cache(&self) -> NArrayCache {
        let budget = VirtualBlockBudget {
            budget: None,
            prev_request_id: None,
        };
        (HashSet::new(), narray::build(&self.dimension, budget))
    }

    fn compute_block_constraints<M: SegmentBudget>(
        &self,
        request_history: Vec<&Request>,
        block: &Block,
        narray_cache: Option<&mut NArrayCache>
    ) -> BlockConstraints<M> {

        // feature flag by env var
        // let cache_requests = std::env::var("CACHE_REQUEST_COST").unwrap_or("0".to_string());

        let mut runtime_measurements: Vec<RuntimeMeasurement> = Vec::new();

        let mut remaining_budget: NArray<VirtualBlockBudget> = if let Some(narray_cache) = narray_cache {
            // let tmp_dir = std::env::temp_dir();
            // if !tmp_dir.exists() {
            //     fs::create_dir_all(&tmp_dir).expect("Failed to create temp directory");
            // }
            // let blkid = block.id;
            // let tmp_file_path = tmp_dir.join(format!("cum_request_cost_{blkid}.bin"));
            // let mut file = File::open(&tmp_file_path);
            // println!("Cache file {:?}", &tmp_file_path);

            // if file.is_ok() {
            //     println!("Cache file size: {:?}", file.as_ref().unwrap().metadata().unwrap().len());
            // }
            //
            // let budget = VirtualBlockBudget {
            //     budget: None,
            //     prev_request_id: None,
            // };
            // let config = config::standard();
            //

            let mut rm_loadcache = RuntimeMeasurement::start(RuntimeKind::LoadCache);


            let (ref mut applied_requests, ref mut cum_request_cost) = narray_cache;
            // Load from disk
            // let (mut applied_requests, mut cum_request_cost): (HashSet<RequestId>, NArray<VirtualBlockBudget>) = if file.is_ok() {
            //     bincode::decode_from_std_read(&mut file.unwrap(), config).expect("Failed to read cache file")
            //     // bincode::deserialize_from(file.unwrap()).expect("Failed to read cache file")
            // } else {
            //     println!("Initializing new narray");
            //     (HashSet::new(), narray::build(&self.dimension, budget))
            // };
            runtime_measurements.push(rm_loadcache.stop());

            println!("  cum_request_cost.size()={:?}", cum_request_cost.size());

            let mut rt_apply = RuntimeMeasurement::start(RuntimeKind::ApplyHistory);
            // Update cum_request_cost with request history
            let mut apply_count = 0;
            for request in request_history.iter() {
                let request = *request;
                if !applied_requests.contains(&request.request_id) {
                    for vec in request.dnf().repeating_iter(self.schema) {
                        let idx = Index::new(&vec);
                        cum_request_cost.update(&idx, request, |vblock, request| {
                            vblock.add(request.request_id, request.request_cost(&block.privacy_unit));
                        });
                        apply_count += 1;
                    }
                    applied_requests.insert(request.request_id);
                }
            }
            runtime_measurements.push(rt_apply.stop());
            println!("Applied {} requests", apply_count);
            // assert we have applied all requests in request_history by comparing all keys in the sets
            // assert!(request_history.iter().map(|r| r.request_id).collect::<HashSet<_>>() == applied_requests., "Not all requests were applied");


            println!("Applied last round's history");

            let mut rt_save = RuntimeMeasurement::start(RuntimeKind::SaveCache);
            // Save to disk or clone
            // let mut file = File::create(&tmp_file_path).expect("Failed to create cache file");
            // bincode::serialize_into(file, &(applied_requests.clone(), &cum_request_cost)).expect("Failed to write cache file");
            // bincode::encode_into_std_write(&(applied_requests.clone(), &cum_request_cost), &mut file, config).expect("Failed to write cache file");

            runtime_measurements.push(rt_save.stop());
            // println!("Saved cache file");

            let mut rt_apply_budget = RuntimeMeasurement::start(RuntimeKind::ApplyBudget);
            // Apply budget on cum_request_cost -> remaining budget
            // We re-use the memory for cum_request_cost as remaining_budget
            let mut remaining_budget = cum_request_cost.clone();

            let update_budget = |vblock: &mut VirtualBlockBudget, budget: &AccountingType| {
                // assert!(vblock.budget.as_ref().unwrap().approx_eq(budget, F64Margin::default()), "Two different budgets for the same virtual block");
                // let tmp = vblock.budget.clone().expect("Block doesnt have budget!");
                // vblock.budget = Some(budget.clone());
                // vblock.update(RequestId(0), &tmp); // subtracts budget from cum_request_cost


                let cur_budget = vblock.budget.as_ref();
                if cur_budget.is_none() {
                    vblock.budget = Some(budget.clone());
                } else {
                    vblock.budget =  Some(budget - cur_budget.unwrap());
                    if vblock.prev_request_id.is_none() {
                        panic!("A block cannot have been selected by no request yet still have a budget. \
                        This probably means its being visited twice!");
                    }
                }
                vblock.prev_request_id = None;
                // TODO: For correctness, ist here a risk we repeat this? --> What does repeating_iter do?
            };
            block.budget_by_section.iter().for_each(|section| {
                for virtual_block_id in section.dnf().repeating_iter(self.schema) {
                    let idx = Index::new(&virtual_block_id);
                    remaining_budget.update(&idx, &section.unlocked_budget, update_budget);
                }
            });
            runtime_measurements.push(rt_apply_budget.stop());

            println!("Updated budget by section");

            remaining_budget
        } else {


            // ==== OLD ====
            let budget = match &block.default_unlocked_budget {
                Some(budget) => {
                    VirtualBlockBudget {
                        budget: Some(budget.clone()),
                        prev_request_id: None,
                    }
                }
                None => {
                    VirtualBlockBudget {
                        budget: None,
                        prev_request_id: None,
                    }
                }
            };

            // TODO [nku] [later]: as alternative could also remove cost / budget from virtualblock and only focus on hash id.
            // afterwards, compute unique hash id and reverse all, then compute for each id cost.
            // also for request history could do this segmentation with hash id.
            // (would potentially save a lot of duplicated computation on adding up accounting_types)

            // Start with default budget for each block
            let mut remaining_budget = narray::build(&self.dimension, budget);

            // Set custom budget from budget_by_section
            let update_budget = |vblock: &mut VirtualBlockBudget, budget: &AccountingType| {
                assert!(vblock.budget.is_none() || vblock.budget.as_ref().unwrap().approx_eq(budget, F64Margin::default()), "Two different budgets for the same virtual block");
                vblock.budget = Some(budget.clone());
            };

            let mut rt_apply = RuntimeMeasurement::start(RuntimeKind::ApplyBudget);

            block.budget_by_section.iter().for_each(|section| {
                for virtual_block_id in section.dnf().repeating_iter(self.schema) {
                    let idx = Index::new(&virtual_block_id);
                    remaining_budget.update(&idx, &section.unlocked_budget, update_budget);
                }
            });
            runtime_measurements.push(rt_apply.stop());
            // TODO [hly]: The post segmentation time of the first round should tell you how long the above budget setting code takes.

            // TODO [hly]: We could keep a cache of <Set<RequestId>, NArray<AccountingType>> for the cum_request_cost
            //   1. Update the cum_request_cost with the request history (essentially applying the costs of the prev round)
            //   2. There are two options based on how much you want to keep in memory:
            //         2.1 Saving Memory:   (requires keeping 2 NArrays in memory and write one to disk)
            //              - You write the cum_request_cost to disk after this update again.
            //              - Writing to disk can be optional, you can just clone it and work on the clone.
            //              - You apply the budget on the "cum_request_cost" itself: value = budget - cum_request_cost
            //              - This now becomes the "remaining_budget"
            //         2.2 Ignoring Memory: (requires keeping 3 NArrays in memory)
            //              - You compute the "budget" ("remaining_budget" called above) with the code above.
            //              - In `calculate_remaining_budget_per_segment(..)` you pass in 3x an narray: cum_request_cost, requested_budget, available_budget (prev name: remaining_budget)
            //              - You iterate over the `requested_budget` array as currently, and then you get an idx. With this idx you lookup the cum_request_cost and the available_budget and "update" the segment with the difference.

            let mut rt_history = RuntimeMeasurement::start(RuntimeKind::ApplyHistory);
            calculate_remaining_budget(&request_history, &mut remaining_budget, block, self.schema);
            runtime_measurements.push(rt_history.stop());

            remaining_budget
        };

        // calculate min remaining budget per segment
        let default_budget = match &block.default_unlocked_budget {
            Some(budget) => {
                VirtualBlockBudget {
                    budget: Some(budget.clone()),
                    prev_request_id: None,
                }
            }
            None => {
                VirtualBlockBudget {
                    budget: None,
                    prev_request_id: None,
                }
            }
        };

        let mut rt_segment = RuntimeMeasurement::start(RuntimeKind::CalculateBySegment);
        // TODO: Are negative budgets considered properly in the rdp_opt_budget consolidation?
        let mut budget_by_segment: HashMap<u64, SegmentWrapper<M>> = HashMap::new();
        calculate_remaining_budget_per_segment(
            &mut budget_by_segment,
            &remaining_budget,
            &self.requested_budget,
            &default_budget,
        );
        runtime_measurements.push(rt_segment.stop());

        // print runtime_measurements
        for rm in runtime_measurements.iter() {
            println!("{:?}", rm);
        }

        // TODO [nku]: Ideally this selection should happen outside of the composition
        let selected_requests: Vec<&Request> = self.request_batch.iter().filter(|req| is_selected_block(req, block)).map(|r| *r).collect();



        // reconstruct request ids from segment id with first_index
        reconstruct_request_ids(&mut budget_by_segment, &selected_requests, block);


        budget_by_segment.retain(|_, v| v.is_contested());


        // find rejected request ids (r.cost > budget)
        //  + remove cost of rejected request ids from cost sums
        //  + retain only congested (after subtraction of rejected requests)
        let rejected_request_ids =
            reject_infeasible_requests(&mut budget_by_segment, &self.request_batch, block);

        // find accepted (all \ rejected \ congested)
        build_block_constraints(
            &rejected_request_ids,
            budget_by_segment,
            &self.request_batch,
        )
    }
}

fn calculate_remaining_budget(
    request_history: &[&Request],
    remaining_budget: &mut NArray<VirtualBlockBudget>,
    block: &Block,
    schema: &Schema,
) {
    let update_budget_closure = |virtual_block_budget: &mut VirtualBlockBudget,
                                 request: &Request| {
        virtual_block_budget.update(request.request_id, request.request_cost(&block.privacy_unit));
    };

    // TODO [hly]: I think here it's possible to have a set of request ids associated with the "cached" cum_request_cost.
    //             Then based on the request history you are getting, you can check which request costs you still need to add to the cum_request_cost.
    //              (Plus maybe also check that there are no additional request ids that are not part of the history accumulated in the cum_request_cost)
    // Set<RequestId>     cum_request_cost: NArray<...>

    for request in request_history.iter() {
        let request = *request;
        for vec in request.dnf().repeating_iter(schema) {
            let idx = Index::new(&vec);
            remaining_budget.update(&idx, request, update_budget_closure);
        }
    }
}

fn calculate_requested_budget(
    request_batch: &[&Request],
    requested_budget: &mut NArray<VirtualBlockRequested>,
    schema: &Schema,
) {
    let update_virtual_block = |block: &mut VirtualBlockRequested, request: &Request| {
        block.update(request.request_id);
    };

    for request in request_batch.iter() {
        let request = *request;
        for vec in request.dnf().repeating_iter(schema) {
            let idx = Index::new(&vec);
            requested_budget.update(&idx, request, update_virtual_block);
        }
    }
}


pub trait AdpCost {
    fn compute_idx_per_segment(&self) -> HashMap<u64, Index>;
    fn max_adp_cost(&self, segments: &HashMap<u64, Index>, relevant_requests: Vec<&Request>, privacy_unit: PrivacyUnit, alphas: &RdpAlphas, delta: f64) -> AccountingType;
}

impl AdpCost for NArraySegmentation<'_, '_>{

    fn compute_idx_per_segment(&self) -> HashMap<u64, Index>{
        let mut segments = HashMap::new();
        for (i, virtual_block_requested_cost) in self.requested_budget.iter().enumerate() {
            segments
                .entry(virtual_block_requested_cost.request_hash)
                .or_insert_with(|| {
                        narray::from_idx(i, &self.requested_budget.dim)
                });
        }
        println!("  n_segments={:?}", segments.len());
        segments
    }


    fn max_adp_cost(&self, segments: &HashMap<u64, Index>, relevant_requests: Vec<&Request>, privacy_unit: PrivacyUnit, alphas: &RdpAlphas, delta: f64) -> AccountingType {

        let budget_type = AccountingType::zero_clone(&self.schema.accounting_type);

        let max_epsilon = segments.iter().map(|(_id, first_idx)| {

            // reconstruct request ids from segment with first_index and sum up their costs
            let (_request_ids, sum_requested_cost) = relevant_requests
                .iter()
                .filter(|r| {
                    // keep only requests which have one co ·─łnjunction that contains the first_idx
                    r.dnf()
                        .conjunctions
                        .iter()
                        .any(|conj| conj.contains(first_idx))
                })
                .map(|r| (r.request_id, r.request_cost(&privacy_unit)))
                .fold((Vec::new(), budget_type.clone()), |(mut ids, mut sum), (id, cost)| {
                    ids.push(id);
                    sum += cost;
                    (ids, sum)
                });

            let adp = sum_requested_cost.rdp_to_adp(alphas, delta);

            let epsilon = match adp {
                AccountingType::EpsDeltaDp { eps , delta: _} => {
                    eps
                }
                _ => panic!("Must be EpsDeltaDp")

            };
            epsilon
        }).max_by(|x, y| x.partial_cmp(y).unwrap());

        AccountingType::EpsDeltaDp { eps: max_epsilon.unwrap() , delta}
    }

}

// could be alternative approach for "calculate_remaining_budget_per_segment"
//fn compute_segments(requests: &[Request], schema: &Schema) -> Vec<Segment> {
//    let dimension: Vec<usize> = schema.attributes.iter().map(|attr| attr.len()).collect();
//    let dimension = Dimension::new(&dimension);
//
//    // calculate requested budget
//    let block = VirtualBlockRequested::new(schema);
//    let mut requested_budget = narray::build(&dimension, block);
//    calculate_requested_budget(requests, &mut requested_budget, schema);
//
//    // reduce to segments
//    let segments = requested_budget
//        .iter()
//        .enumerate()
//        .into_grouping_map_by(|(_i, virtual_block)| virtual_block.request_hash)
//        .aggregate(|acc: Option<Segment>, request_hash, (i, virtual_block)| {
//            match acc {
//                Some(Segment {
//                    id: _,
//                    request_ids: _,
//                    accounting: SegmentAccounting::Cost(cost),
//                }) => {
//                    cost += &virtual_block.cost;
//                    // TODO [nku] [later] verify that the change was done on return value
//                    acc
//                }
//                None => Some(Segment {
//                    id: *request_hash,
//                    request_ids: None,
//                    accounting: SegmentAccounting::Cost(virtual_block.cost),
//                }),
//                _ => panic!("illegal x"),
//            }
//        });
//
//    // TODO [nku] [later] could also think about returning HashMap of Segments
//    segments.into_iter().map(|(k, v)| v).collect()
//}

fn calculate_remaining_budget_per_segment<M: SegmentBudget>(
    budget_by_segment: &mut HashMap<u64, SegmentWrapper<M>>,
    remaining_budget: &NArray<VirtualBlockBudget>,
    requested_budget: &NArray<VirtualBlockRequested>,
    default_budget: &VirtualBlockBudget
) {
    for (i, virtual_block_requested_cost) in requested_budget.iter().enumerate() {
        let virtual_block_budget = remaining_budget.get_by_flat(i);

        budget_by_segment
            .entry(virtual_block_requested_cost.request_hash)
            .or_insert_with(|| {
                SegmentWrapper::new(
                    virtual_block_requested_cost.request_hash,
                    narray::from_idx(i, &requested_budget.dim),
                )
            })
            .update_segment(virtual_block_budget, virtual_block_requested_cost, default_budget);
    }
}

trait Contains {
    fn contains(&self, idx: &Index) -> bool;
}

impl Contains for Conjunction {
    fn contains(&self, idx: &Index) -> bool {
        let idx_vec = idx.to_vec();

        if idx_vec.len() != self.predicates().len() {
            panic!(
                "incompatible index for conjunction idx_vec={:?}   pred length={:?}",
                idx_vec.len(),
                self.predicates().len()
            );
        }

        self.predicates()
            .iter()
            .zip(idx_vec.iter())
            .all(|(pred, i)| pred.contains(i))
    }
}

fn reconstruct_request_ids<M: SegmentBudget>(
    segments: &mut HashMap<u64, SegmentWrapper<M>>,
    requests: &[&Request],
    block: &Block,
) {

    let budget_type = match &block.default_total_budget {
        Some(budget) => AccountingType::zero_clone(budget),
        None =>  AccountingType::zero_clone(&block.budget_by_section.iter().next().unwrap().total_budget),
    };

    for (_id, segment_wrapper) in segments.iter_mut() {
        let first_idx = &segment_wrapper.first_idx;

        let (request_ids, sum_requested_cost) = requests
            .iter()
            .filter(|r| {
                // keep only requests which have one conjunction that contains the first_idx
                r.dnf()
                    .conjunctions
                    .iter()
                    .any(|conj| conj.contains(first_idx))
            })
            .map(|r| (r.request_id, r.request_cost(&block.privacy_unit)))
            .fold((Vec::new(), budget_type.clone()), |(mut ids, mut sum), (id, cost)| {
                ids.push(id);
                sum += cost;
                (ids, sum)
            });


        segment_wrapper.sum_requested_cost = Some(sum_requested_cost);
        segment_wrapper.segment.request_ids = Some(request_ids);

    }
}

fn reject_infeasible_requests<M: SegmentBudget>(
    segments: &mut HashMap<u64, SegmentWrapper<M>>,
    requests: &[&Request],
    block: &Block,
) -> HashSet<RequestId> {
    let request_cost_map: HashMap<RequestId, &AccountingType> = requests
        .iter()
        .map(|r| (r.request_id, r.request_cost(&block.privacy_unit)))
        .collect();

    let mut rejected_request_ids: HashSet<RequestId> = HashSet::new();

    for (_id, segment_wrapper) in segments.iter() {
        // loop over request_ids -> see if r.cost > segment.remaining_budget -> if yes => remove from request_ids, subtract cost from sum_requested_cost, put into rejected ids list
        let request_ids = segment_wrapper
            .segment
            .request_ids
            .as_ref()
            .expect("must be there: request ids");

        let iter = request_ids
            .iter()
            .filter(|request_id| {
                let request_cost = *request_cost_map
                    .get(*request_id)
                    .expect("must be there: request_cost_map");

                let is_budget_sufficient = segment_wrapper
                    .segment
                    .remaining_budget
                    .is_budget_sufficient(request_cost);

                !is_budget_sufficient
            })
            .copied();

        rejected_request_ids.extend(iter);
    }

    // after identifying rejected requests -> need to update all segments
    if !rejected_request_ids.is_empty() {
        segments.retain(|_id, segment_wrapper| {
            // remove rejected requests from segment and adapt request cost sum

            match &mut segment_wrapper.segment.request_ids {
                Some(request_ids) => request_ids.retain(|request_id| {
                    let contains = rejected_request_ids.contains(request_id);

                    if contains {
                        // segment contains rejected request => subtract rejected request cost from sum

                        if let Some(sum_requested_cost) = &mut segment_wrapper.sum_requested_cost {
                            let cost = *request_cost_map.get(request_id).unwrap();
                            *sum_requested_cost -= cost;
                        } else {
                            panic!("sum request cost must be set");
                        }
                    }

                    !contains
                }),
                None => panic!("request ids must be defined"),
            }

            segment_wrapper.is_contested()
        });
    }

    rejected_request_ids
}

fn build_block_constraints<'a, M: SegmentBudget>(
    rejected_request_ids: &HashSet<RequestId>,
    contested_segments: HashMap<u64, SegmentWrapper<M>>,
    request_batch: &[&'a Request],
) -> BlockConstraints<M> {
    let request_map: HashMap<RequestId, &Request> =
        request_batch.iter().map(|r| (r.request_id, *r)).collect();

    /*
    println!("request_map={:?}", request_map.keys());
    println!("contested_requests={:?}", contested_segments.keys());
    */

    let contested: HashSet<RequestId> = contested_segments
        .values()
        .flat_map(|segment_wrapper| {
            segment_wrapper
                .segment
                .request_ids
                .as_ref()
                .expect("request ids must be set")
        })
        .copied()
        .collect();

    let acceptable = request_batch
        .iter()
        .filter(|request| {
            (!rejected_request_ids.contains(&request.request_id))
                && (!contested.contains(&request.request_id))
        })
        .map(|r| r.request_id)
        .collect();

    let rejected = rejected_request_ids
        .iter()
        .map(|r_id| *request_map.get(r_id).expect("unknown request"))
        .map(|r| r.request_id)
        .collect();

    // TODO [nku] THIS can be replaced with own implmentation of problemformulation (or new BlockCOnstraints)
    let contested_segments = contested_segments
        .into_iter()
        .map(|(_segment_id, segment_wrapper)| segment_wrapper.segment)
        .into_grouping_map_by(|segment| {
            let mut hasher = DefaultHasher::new();
            Hash::hash_slice(segment.request_ids.as_ref().unwrap(), &mut hasher);
            hasher.finish()
        })
        .fold_first(|mut acc_segment, _key, val_segment| {
            // merge both of them
            acc_segment
                .remaining_budget
                .merge_assign(&val_segment.remaining_budget);

            acc_segment
        })
        .into_values()
        .filter(|segment| !segment.request_ids.as_ref().unwrap().is_empty())
        .collect_vec();

    BlockConstraints {
        acceptable,
        rejected,
        contested,
        contested_segments,
    }
}

// TODO [nku] [later] test with better hash function
/*
const K: usize = 0x517cc1b727220a95;
fn fx_hasher(start: usize, new: usize) -> usize {
    let tmp = start.rotate_left(5) ^ new;
    tmp.wrapping_mul(K)
}
 */

impl VirtualBlockBudget {
    fn update(&mut self, request_id: RequestId, privacy_cost: &AccountingType) {
        match self.prev_request_id {
            Some(prev_request_id) if prev_request_id == request_id => (), // do nothing
            // ignore (because we use repeating iter -> can happen that we select same block twice)
            _ => {
                // subtract the privacy cost from the budget and update request id to ensure that we only do it once
                self.prev_request_id = Some(request_id);
                //let mut cur_budget = self.budget.expect("budget must be set before updating");

                let cur_budget = self.budget.as_ref().expect("budget must be set before updating");
                self.budget =  Some(cur_budget - privacy_cost);
            }
        }
    }
    fn add(&mut self, request_id: RequestId, privacy_cost: &AccountingType) {
        match self.prev_request_id {
            Some(prev_request_id) if prev_request_id == request_id => (), // do nothing
            // ignore (because we use repeating iter -> can happen that we select same block twice)
            _ => {
                // subtract the privacy cost from the budget and update request id to ensure that we only do it once
                self.prev_request_id = Some(request_id);

                // Init at zero
                let cur_budget = self.budget.as_ref();
                if let Some(cur_budget) = cur_budget {
                    self.budget =  Some(cur_budget + privacy_cost);
                } else {
                    self.budget = Some(privacy_cost.clone());
                }
            }
        }
    }
    fn is_budget_applied(&self) -> bool {
        self.prev_request_id.is_none() && self.budget.is_some()
    }
}

// TODO [nku] [later]: The hash function is not the bottleneck -> with a better hash function we can ignore request_count logic
impl VirtualBlockRequested {
    fn new(_schema: &Schema) -> VirtualBlockRequested {
        VirtualBlockRequested {
            request_hash: 0,
            request_count: 0,
            prev_request_id: None,
        }
    }

    fn update(&mut self, request_id: RequestId) {
        match self.prev_request_id {
            Some(prev_request_id) if prev_request_id == request_id => (), // do nothing if prev request id is the same (deal with repeating iter)
            _ => {
                // TODO [nku] [later] could bring back request_hash with fx_hasher(...) -> Problem can observe hash collisions
                //self.request_hash = fx_hasher(self.request_hash, request_id);

                self.request_hash = Hash64::hash_with_seed(
                    request_id.0.to_ne_bytes(),
                    (self.request_hash, 0, 0, 0),
                );

                self.request_count += 1;
                self.prev_request_id = Some(request_id);
            }
        }
    }
}

pub struct SegmentWrapper<M: SegmentBudget> {
    segment: BlockSegment<M>,
    first_idx: Index,
    sum_requested_cost: Option<AccountingType>,
    request_count: Option<u32>,
}

impl<M: SegmentBudget> SegmentWrapper<M> {
    fn new(segment_id: u64, first_idx: Index) -> SegmentWrapper<M> {
        SegmentWrapper {
            segment: BlockSegment::new(segment_id.try_into().unwrap()),
            first_idx,
            sum_requested_cost: None,
            request_count: None,
        }
    }

    // at the moment also for rdp we return one budget per segment (the element-wise minimum)
    fn update_segment(
        &mut self,
        virtual_block_budget: &VirtualBlockBudget,
        virtual_block_requested_cost: &VirtualBlockRequested,
        default_budget: &VirtualBlockBudget,
    ) {
        // TODO: Discuss this logic
        if virtual_block_budget.budget.is_some() {
            self.segment
                .remaining_budget
                .add_budget_constraint(virtual_block_budget.budget.as_ref().expect("budget must be set"));
        } else {
            self.segment
                .remaining_budget
                .add_budget_constraint(default_budget.budget.as_ref().expect("default budget must be set"));
        }

        // 5914157414052549729

        match self.request_count {
            // set the cost if not set previously
            None => self.request_count = Some(virtual_block_requested_cost.request_count),

            // assert that n_requests is the same for all blocks in segment
            Some(request_count) => {
                assert_eq!(request_count, virtual_block_requested_cost.request_count)
            } // TODO [nku] [later] want some assert which is not present in --release
        }

        // -> removed because we now only sum up the costs after reconstructing the request ids
//        match &self.sum_requested_cost {
//            // set the cost if not set previously
//            None => self.sum_requested_cost = Some(virtual_block_requested_cost.cost.clone()),
//
//            // assert that cost is the same for all blocks in segment
//            Some(sum_requested_cost) => assert!(
//                sum_requested_cost
//                    .approx_eq(&virtual_block_requested_cost.cost, F64Margin::default()),
//                "segment_id={}     my_sum={:?}  other_sum={:?}",
//                self.segment.id,
//                sum_requested_cost,
//                virtual_block_requested_cost.cost
//            ),
//            // TODO [nku] [later] want some assert which is not present in --release
//        }
    }

    fn is_contested(&self) -> bool {
        let cost = self
            .sum_requested_cost
            .as_ref()
            .expect("requested cost must be set");

        !self.segment.remaining_budget.is_budget_sufficient(cost)
    }
}

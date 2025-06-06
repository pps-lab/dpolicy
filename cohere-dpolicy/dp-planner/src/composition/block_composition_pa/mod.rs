use crate::block::{Block, BlockId};
use crate::composition::block_composition_pa::algo_narray::{narray, NArraySegmentation};
use crate::composition::ProblemFormulation;
use crate::config::SegmentationAlgo;
use crate::dprivacy::budget::SegmentBudget;
use crate::logging::{RuntimeKind, RuntimeMeasurement};
use crate::request::{Request, RequestId};
use crate::schema::Schema;
use crate::util;
use rayon::prelude::*;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt::Debug;
use std::ops::DerefMut;
use std::sync::{Arc, Mutex, RwLock};
use log::debug;
use std::time::Instant;
use crate::composition::block_composition_pa::algo_narray::narray::NArray;
use super::{BlockConstraints, CompositionConstraint};

pub mod algo_narray;

pub struct BlockCompositionPartAttributes {
    pub config: SegmentationAlgo,
    pub num_threads: Option<usize>,
}

pub(crate) type NArrayCache = (HashSet<RequestId>, crate::composition::block_composition_pa::algo_narray::narray::NArray<crate::composition::block_composition_pa::algo_narray::VirtualBlockBudget>);

pub fn build_block_part_attributes(config: SegmentationAlgo, num_threads: Option<usize>) -> BlockCompositionPartAttributes {
    BlockCompositionPartAttributes { config, num_threads}
}

impl CompositionConstraint for BlockCompositionPartAttributes {
    fn build_problem_formulation<M: SegmentBudget>(
        &self,
        blocks: &HashMap<BlockId, Block>,
        candidate_requests: &BTreeMap<RequestId, Request>,
        history_requests: &HashMap<RequestId, Request>,
        schema: &Schema,
        runtime_measurements: &mut Vec<RuntimeMeasurement>,
        block_narray_cache: &mut HashMap<BlockId, Arc<RwLock<NArrayCache>>>
    ) -> super::ProblemFormulation<M> {
        // TODO [later] request needs concept for specifying which blocks are interesting (can then be used for DPF and also for UserTimeBlocking)

        let request_batch: Vec<&Request> = candidate_requests.values().collect();

//        // blocks which have either different history or different budget
//        let mut unique_budget_and_history: Vec<(AccountingType, BTreeSet<RequestId>, &Block)> =
//            Vec::new();
//        // for non-unique blocks, store which block is the one which is the same
//        let mut lookup: HashMap<BlockId, BlockId> = HashMap::new();
//        'outer: for block in blocks.values() {
//            let history_set = BTreeSet::from_iter(block.request_history.iter().copied());
//            // only want to add block, if budget is not approx eq to existing budget or history
//            // is different
//
//            // FIXME [nku] ONLY LOOKING AT THE DEFAULT BUDGET IS NOT SUFFICIENT ANYMORE -> JUST TAKE ALL BLOCKS
//                  -> ALTERNATIVELY WE WOULD ALSO NEED TO CHECK THAT THE SECTIONS ARE THE SAME AND THE PRIVACY UNIT OF THE BLOCK IS THE SAME
//
//            for (budget, history, other_block) in unique_budget_and_history.iter() {
//                if (&block.default_unlocked_budget).approx_eq(budget, F64Margin::default())
//                    && history == &history_set
//                {
//                    lookup.insert(block.id, other_block.id);
//                    continue 'outer;
//                }
//            }
//            unique_budget_and_history.push((block.default_unlocked_budget.clone(), history_set, block))
//        }

        let num_threads = self.num_threads.unwrap_or_else(num_cpus::get);
        assert!(
            num_threads > 0 && num_threads <= num_cpus::get(),
            "num_threads must be greater than 0 and smaller than or equal \
        to the number of virtual cores ({}) (given number of threads was {})",
            num_cpus::get(),
            num_threads
        );

        let cache_requests = std::env::var("CACHE_REQUEST_COST").unwrap_or("0".to_string()) == "1";

        // Clear unused block cache
        if cache_requests {
            block_narray_cache.retain(|k, _| blocks.contains_key(k));
        }
        let block_narray_cache = Arc::new(RwLock::new(block_narray_cache));

        let unique_block_constraints = util::create_pool(num_threads)
            .expect("Failed to create thread pool")
            .install(|| {

                let map = blocks
                    .into_par_iter()
                    .map(|(_, block)| {
                        let request_history: Vec<&Request> = block
                            .request_history
                            .iter()
                            .map(|r_id| {
                                history_requests
                                    .get(r_id)
                                    .expect("request id not in history")
                            })
                            .collect();

                        (block, request_history)
                    });

                println!(
                    "rayon threads = {:?}    blocks to process = {:?}",
                    rayon::current_num_threads(),
                    blocks.len()
                );
                let start = Instant::now();

                let unique_block_constraints: HashMap<BlockId, BlockConstraints<M>> = match self.config {
                    SegmentationAlgo::Narray{..} => {
                        let mut seg_meas = RuntimeMeasurement::start(RuntimeKind::Segmentation);
                        let segmentation = NArraySegmentation::new(request_batch, schema);
                        runtime_measurements.push(seg_meas.stop());

                        let mut seg_post_meas = RuntimeMeasurement::start(RuntimeKind::PostSegmentation);
                        let constraints = map
                            .map(|(block, request_history)| {
                                if (cache_requests) {
                                    let map = block_narray_cache.read().expect("Failed to read block narray cache");

                                    if let Some(cache) = map.get(&block.id) {
                                        println!("Using cache for block {}", block.id);
                                        (
                                            block.id,
                                            segmentation
                                                .compute_block_constraints(request_history, &block, Some(cache.write().as_deref_mut().expect("Failed to get mutable reference to cache"))),
                                        )
                                    } else {
                                        // Insert new cache
                                        println!("Did not find cache for block {}", block.id);

                                        drop(map);

                                        // Run on fresh cache
                                        let mut fresh_cache = segmentation.new_cache();
                                        let constraints = segmentation
                                            .compute_block_constraints(request_history, &block, Some(&mut fresh_cache));

                                        let mut map = block_narray_cache.write().expect("Failed to write block narray cache");
                                        map.insert(block.id, Arc::new(RwLock::new(fresh_cache)));
                                        println!("Storing cache for block {}", block.id);

                                        (
                                            block.id,
                                            constraints
                                        )
                                    }
                                } else {
                                    (
                                        block.id,
                                        segmentation
                                            .compute_block_constraints(request_history, &block, None),
                                    )
                                }

                                //
                                // let default = Arc::new(RwLock::new(segmentation.new_cache()));
                                // let mut write_guard = block_narray_cache.write().expect("Failed to write block narray cache");
                                // write_guard.insert(block.id, default);
                                //
                                // let cache = block_narray_cache.read().expect("Failed to read block narray cache")
                                //     .get(&block.id).unwrap(); // safe to unwrap, as we just inserted it
                                // let mut write_cache = cache.write().expect("Failed to write block narray cache");
                                //
                                // // .unwrap_or_else(|| {
                                // //
                                // //     });
                                //
                                // (
                                //     block.id,
                                //     segmentation
                                //         .compute_block_constraints(request_history, &block, Some(write_cache.deref_mut())),
                                // )
                            })
                            .collect();

                        // loop so we can capture narray_cache // TODO: Remove
                        // let mut constraints: HashMap<BlockId, BlockConstraints<M>> = HashMap::new();
                        // for (block, request_history) in map {
                        //     constraints.insert(block.id, segmentation.compute_block_constraints(request_history, &block, &mut narray_cache));
                        // }
                        runtime_measurements.push(seg_post_meas.stop());

                        constraints
                    }
                };

                debug!(
                    "rayon threads = {:?}    elapsed hist={:?}",
                    rayon::current_num_threads(),
                    start.elapsed()
                );

                unique_block_constraints
            });

        // NOTE: As we don't process only the unique blocks anymore, we don't need the lookup here.
        let block_constraints = unique_block_constraints;
        //let block_constraints = blocks
        //    .iter()
        //    .map({
        //        |(bid, _)| {
        //            if unique_block_constraints.contains_key(bid) {
        //                (*bid, unique_block_constraints[bid].clone())
        //            } else {
        //                (*bid, unique_block_constraints[&lookup[bid]].clone())
        //            }
        //        }
        //    })
        //    .collect();

        ProblemFormulation::new(block_constraints, candidate_requests, blocks)
    }
}

// TODO [nku] question: can we remove lifetimes from segmentation? internally request_batch might not need to be stored, schema can also be passed when needed
// TODO [nku] question: can we have incremental solution? basically, that we don't have to re-apply complete history every time? ~don't want to store the whole thing for every block forever
pub trait Segmentation<'r, 's> {
    fn new(request_batch: Vec<&'r Request>, schema: &'s Schema) -> Self;

    fn new_cache(&self) -> NArrayCache;

    fn compute_block_constraints<M: SegmentBudget + Debug>(
        &self,
        request_history: Vec<&Request>,
        block: &Block,
        narray_cache: Option<&mut NArrayCache>
    ) -> BlockConstraints<M>;
}

#[cfg(test)]
mod tests {

    use float_cmp::{ApproxEq, F64Margin};

    use crate::block::{Block, BlockId};
    use crate::composition::{BlockConstraints, BlockSegment};
    use crate::config::BudgetTotal;
    use crate::dprivacy::budget::{OptimalBudget, SegmentBudget};
    use crate::dprivacy::privacy_unit::PrivacyUnit;
    use crate::dprivacy::{AccountingType, AdpAccounting};

    use crate::request::{
        load_requests, resource_path, AttributeId, ConjunctionBuilder, Predicate, Request, RequestBuilder, RequestId
    };
    use crate::schema::{load_schema, Attribute, Schema, ValueDomain};

    use crate::simulation::RoundId;
    use crate::dprivacy::AccountingType::EpsDeltaDp;

    use crate::dprivacy::rdp_alphas_accounting::RdpAlphas::*;
    use crate::util::{CENSUS_REQUESTS, CENSUS_SCHEMA};
    use itertools::Itertools;
    use std::collections::{HashMap, HashSet};
    use std::time::Instant;

    use super::algo_narray::NArraySegmentation;
    use super::Segmentation;

    #[test]
    fn test_narray_segmentation_dummy() {
        let schema = build_dummy_schema();

        let requests = build_dummy_requests(&schema);
        let request_history = &requests[0..2];
        let request_batch = &requests[2..6];

        let algo = NArraySegmentation::new(request_batch.iter().collect(), &schema);

        let block = build_dummy_block(AccountingType::EpsDp { eps: 1.0 }, request_history);


        let problem: BlockConstraints<OptimalBudget> = algo.compute_block_constraints(
            request_history.iter().collect(),
            &block,
        );

        check_dummy_problem_formulation(problem);
    }


    fn build_dummy_schema() -> Schema {
        // test with two partitioning attributes one with two values and the other with three
        let attributes: Vec<Attribute> = vec![
            Attribute {
                name: "a1".to_owned(),
                value_domain: ValueDomain::Range { min: 0, max: 1 },
                value_domain_map: None,
            },
            Attribute {
                name: "a2".to_owned(),
                value_domain: ValueDomain::Range { min: 0, max: 2 },
                value_domain_map: None,
            },
        ];

        Schema {
            accounting_type: AccountingType::EpsDp { eps: 1.0 },
            attributes,
            name_to_index: HashMap::new(),
            privacy_units: HashSet::from([PrivacyUnit::User]),
            block_sliding_window_size: 1,
        }
    }

    fn build_dummy_block(budget: AccountingType, request_history: &[Request]) -> Block {

        Block {
            id: BlockId(0),
            request_history: request_history.iter().map(|r| r.request_id).collect(),
            default_unlocked_budget: Some(budget.clone()),
            default_total_budget: Some(budget.clone()),
            budget_by_section: Vec::new(),
            privacy_unit: PrivacyUnit::User,
            privacy_unit_selection: None,
            created: RoundId(0),
            retired: None,
        }
    }

    fn build_dummy_requests(schema: &Schema) -> Vec<Request> {
        // with budget 1.0, on each virtual block we can run 2 but not three requests
        let request_cost = AccountingType::EpsDp { eps: 0.4 };
        let request_cost = HashMap::from([(PrivacyUnit::User, request_cost)]);
        let unit_selection = HashMap::new();

        let requests = vec![
            RequestBuilder::new(
                RequestId(0),
                request_cost.clone(),
                unit_selection.clone(),
                1,
                1,
                schema,
            )
            .or_conjunction(
                ConjunctionBuilder::new(schema)
                    .and(AttributeId(1), Predicate::Eq(0))
                    .build(),
            )
            .build(),
            RequestBuilder::new(
                RequestId(1),
                request_cost.clone(),
                unit_selection.clone(),
                1,
                1,
                schema,
            )
            .or_conjunction(
                ConjunctionBuilder::new(schema)
                    .and(AttributeId(0), Predicate::Eq(0))
                    .build(),
            )
            .build(),
            RequestBuilder::new(
                RequestId(2),
                request_cost.clone(),
                unit_selection.clone(),
                1,
                1,
                schema,
            )
            .or_conjunction(
                ConjunctionBuilder::new(schema)
                    .and(AttributeId(0), Predicate::Eq(1))
                    .and(AttributeId(1), Predicate::Eq(2))
                    .build(),
            )
            .build(),
            RequestBuilder::new(
                RequestId(3),
                request_cost.clone(),
                unit_selection.clone(),
                1,
                1,
                schema,
            )
            .or_conjunction(
                ConjunctionBuilder::new(schema)
                    .and(AttributeId(1), Predicate::Between { min: 1, max: 2 })
                    .build(),
            )
            .build(),
            RequestBuilder::new(
                RequestId(4),
                request_cost.clone(),
                unit_selection.clone(),
                1,
                1,
                schema,
            )
            .or_conjunction(
                ConjunctionBuilder::new(schema)
                    .and(AttributeId(1), Predicate::Eq(1))
                    .build(),
            )
            .or_conjunction(
                ConjunctionBuilder::new(schema)
                    .and(AttributeId(0), Predicate::Eq(0))
                    .and(AttributeId(1), Predicate::Eq(2))
                    .build(),
            )
            .build(),
            RequestBuilder::new(
                RequestId(5),
                request_cost,
                unit_selection.clone(),
                1,
                1,
                schema,
            )
            .or_conjunction(
                ConjunctionBuilder::new(schema)
                    .and(AttributeId(0), Predicate::Eq(0))
                    .and(AttributeId(1), Predicate::Between { min: 0, max: 1 })
                    .build(),
            )
            .build(),
        ];

        requests
    }

    fn check_dummy_problem_formulation(problem: BlockConstraints<OptimalBudget>) {
        let actual_rejected_request_ids: HashSet<RequestId> =
            problem.rejected.iter().copied().collect();

        let actual_accepted_request_ids: HashSet<RequestId> =
            problem.acceptable.iter().copied().collect();

        let actual_contested_request_ids: HashSet<RequestId> =
            problem.contested.iter().copied().collect();

        println!(
            "rejected={:?}   accepted={:?}   contested={:?}",
            actual_rejected_request_ids, actual_accepted_request_ids, actual_contested_request_ids
        );

        assert_eq!(
            actual_rejected_request_ids,
            vec![RequestId(5)].into_iter().collect(),
            "rejected ids do not match"
        );

        assert_eq!(
            actual_accepted_request_ids,
            vec![RequestId(2)].into_iter().collect(),
            "accepted ids do not match"
        );

        assert_eq!(
            actual_contested_request_ids,
            vec![RequestId(3), RequestId(4),].into_iter().collect(),
            "contested ids do not much"
        );

        assert_eq!(
            problem.contested_segments.len(),
            1,
            "there is only one contested segment (between r3 and r4)"
        );

        match &problem.contested_segments[0] {
            BlockSegment {
                id: _,
                request_ids: Some(request_ids),
                remaining_budget,
            } => {
                let mut actual = request_ids.clone();
                actual.sort_unstable();
                assert_eq!(
                    actual,
                    vec![RequestId(3), RequestId(4)],
                    "contested request ids not correct on segment"
                );

                assert_eq!(
                    remaining_budget.get_budget_constraints().len(),
                    1,
                    "only one remaining budget"
                );
                assert!(
                    remaining_budget.get_budget_constraints()[0]
                        .approx_eq(&AccountingType::EpsDp { eps: 0.6 }, F64Margin::default()),
                    "remaining budget does not match"
                )
            }
            _ => panic!("illegal segment"),
        }
    }

    #[test]
    fn test_narray_segmentation_dummy_rdp() {
        let schema = build_dummy_schema_rdp();

        let requests = build_dummy_requests_rdp(&schema);
        let request_history = &requests[0..2];
        let request_batch = &requests[2..requests.len()];

        let algo = NArraySegmentation::new(request_batch.iter().collect(), &schema);

        let block = build_dummy_block(AccountingType::Rdp {
            eps_values: A5([3., 0., 0., 0., 3.]),
        }, request_history);

        let problem: BlockConstraints<OptimalBudget> = algo.compute_block_constraints(
            request_history.iter().collect(),
            &block
        );

        check_dummy_problem_formulation_rdp(problem);
    }


    fn build_dummy_schema_rdp() -> Schema {
        // test with two partitioning attributes one with two values and the other with three
        let attributes: Vec<Attribute> = vec![
            Attribute {
                name: "a1".to_owned(),
                value_domain: ValueDomain::Range { min: 0, max: 2 },
                value_domain_map: None,
            },
            Attribute {
                name: "a2".to_owned(),
                value_domain: ValueDomain::Range { min: 0, max: 2 },
                value_domain_map: None,
            },
        ];

        Schema {
            accounting_type: AccountingType::Rdp {
                eps_values: A5([-1., -1., -1., -1., -1.]), // should not use these values
            },
            attributes,
            name_to_index: HashMap::new(),
            privacy_units: HashSet::from([PrivacyUnit::User]),
            block_sliding_window_size: 1,
        }
    }

    fn build_dummy_requests_rdp(schema: &Schema) -> Vec<Request> {
        // with budget [3., 0., 0., 0., 3.] we can run two requests with different costs, but not with the same
        let request_cost1 = AccountingType::Rdp {
            eps_values: A5([2., 1., 1., 1., 1.]),
        };
        let request_cost1 = HashMap::from([(PrivacyUnit::User, request_cost1)]);
        let request_cost2 = AccountingType::Rdp {
            eps_values: A5([1., 1., 1., 1., 2.]),
        };
        let request_cost2 = HashMap::from([(PrivacyUnit::User, request_cost2)]);

        let unit_selection = HashMap::new();

        let requests = vec![
            RequestBuilder::new(
                RequestId(0),
                request_cost1.clone(),
                unit_selection.clone(),
                1,
                1,
                schema,
            )
            .or_conjunction(
                ConjunctionBuilder::new(schema)
                    .and(AttributeId(0), Predicate::Between { min: 0, max: 2 })
                    .and(AttributeId(1), Predicate::Eq(0))
                    .build(),
            )
            .build(),
            RequestBuilder::new(
                RequestId(1),
                request_cost2.clone(),
                unit_selection.clone(),
                1,
                1,
                schema,
            )
            .or_conjunction(
                ConjunctionBuilder::new(schema)
                    .and(AttributeId(0), Predicate::Eq(0))
                    .and(AttributeId(1), Predicate::Between { min: 0, max: 2 })
                    .build(),
            )
            .build(),
            RequestBuilder::new(
                RequestId(2),
                request_cost1.clone(),
                unit_selection.clone(),
                1,
                1,
                schema,
            )
            .or_conjunction(
                ConjunctionBuilder::new(schema)
                    .and(AttributeId(0), Predicate::Between { min: 0, max: 1 })
                    .and(AttributeId(1), Predicate::Between { min: 0, max: 1 })
                    .build(),
            )
            .build(),
            RequestBuilder::new(
                RequestId(3),
                request_cost1.clone(),
                unit_selection.clone(),
                1,
                1,
                schema,
            )
            .or_conjunction(
                ConjunctionBuilder::new(schema)
                    .and(AttributeId(0), Predicate::Between { min: 0, max: 2 })
                    .and(AttributeId(1), Predicate::Eq(1))
                    .build(),
            )
            .build(),
            RequestBuilder::new(
                RequestId(4),
                request_cost2.clone(),
                unit_selection.clone(),
                1,
                1,
                schema,
            )
            .or_conjunction(
                ConjunctionBuilder::new(schema)
                    .and(AttributeId(0), Predicate::Between { min: 1, max: 2 })
                    .and(AttributeId(1), Predicate::Between { min: 1, max: 2 })
                    .build(),
            )
            .build(),
            RequestBuilder::new(
                RequestId(5),
                request_cost1,
                unit_selection.clone(),
                1,
                1,
                schema,
            )
            .or_conjunction(
                ConjunctionBuilder::new(schema)
                    .and(AttributeId(0), Predicate::Between { min: 0, max: 2 })
                    .and(AttributeId(1), Predicate::Between { min: 1, max: 2 })
                    .build(),
            )
            .build(),
            RequestBuilder::new(
                RequestId(6),
                request_cost2,
                unit_selection.clone(),
                1,
                1,
                schema,
            )
            .or_conjunction(
                ConjunctionBuilder::new(schema)
                    .and(AttributeId(0), Predicate::Eq(2))
                    .and(AttributeId(1), Predicate::Eq(0))
                    .build(),
            )
            .build(),
        ];

        requests
    }

    fn check_dummy_problem_formulation_rdp(problem: BlockConstraints<OptimalBudget>) {
        let actual_rejected_request_ids: HashSet<RequestId> =
            problem.rejected.iter().copied().collect();

        let actual_accepted_request_ids: HashSet<RequestId> =
            problem.acceptable.iter().copied().collect();

        let actual_undecided_request_ids: HashSet<RequestId> =
            problem.contested.iter().copied().collect();

        println!(
            "rejected={:?}   accepted={:?}   undecided={:?}",
            actual_rejected_request_ids, actual_accepted_request_ids, actual_undecided_request_ids
        );

        assert_eq!(
            actual_rejected_request_ids,
            vec![RequestId(2)].into_iter().collect(),
            "rejected ids do not match"
        );

        assert_eq!(
            actual_accepted_request_ids,
            vec![RequestId(6)].into_iter().collect(),
            "accepted ids do not match"
        );

        assert_eq!(
            actual_undecided_request_ids,
            vec![RequestId(3), RequestId(4), RequestId(5)]
                .into_iter()
                .collect(),
            "contested ids do not much"
        );

        assert_eq!(
            problem.contested_segments.len(),
            2,
            "there are exactly two contested segments ([r3, r5] and [r3, r4, r5]), but given was {:?}",
            problem
                .contested_segments
                .iter()
                .map(|segment| &segment.request_ids)
                .collect::<Vec<_>>()
        );

        for contested_segment in problem.contested_segments.iter() {
            if contested_segment.request_ids.is_none() {
                panic!("Segment id for contested segment was none")
            } else if contested_segment
                .request_ids
                .as_ref()
                .unwrap()
                .iter()
                .copied()
                .sorted()
                .collect::<Vec<_>>()
                == vec![RequestId(3), RequestId(5)]
            {
                assert!(
                    contested_segment
                        .remaining_budget
                        .get_budget_constraints()
                        .approx_eq(
                            &[&AccountingType::Rdp {
                                eps_values: A5([2., -1., -1., -1., 1.])
                            }],
                            F64Margin::default()
                        ),
                    "Expected: [2., 0., 0., 0., 1.], given {:?}",
                    contested_segment.remaining_budget.get_budget_constraints()
                )
            } else if contested_segment
                .request_ids
                .as_ref()
                .unwrap()
                .iter()
                .copied()
                .sorted()
                .collect::<Vec<_>>()
                == vec![RequestId(3), RequestId(4), RequestId(5)]
            {
                assert!(
                    contested_segment
                        .remaining_budget
                        .get_budget_constraints()
                        .approx_eq(
                            &[&AccountingType::Rdp {
                                eps_values: A5([3., 0., 0., 0., 3.])
                            }],
                            F64Margin::default()
                        ),
                    "Expected: [2., 0., 0., 0., 1.], given {:?}",
                    contested_segment.remaining_budget.get_budget_constraints()
                )
            } else {
                panic!("there are exactly two contested segments ([r3, r5] and [r3, r4, r5]), but given was {:?}",
                       problem
                           .contested_segments
                           .iter()
                           .map(|segment| &segment.request_ids)
                           .collect::<Vec<_>>()
                )
            }
        }
    }

    #[test]
    #[ignore]
    fn test_algorithm_equality_multiple() {
        let epsilons = [2., 4., 8., 16., 32.];
        let deltas = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10];
        let history_frac = 0.1;

        let mut durations = (0u128, 0u128);

        let rdp13 = BudgetTotal{
            privacy_units: Some(vec![PrivacyUnit::User]),
            alphas: Some(vec![0.; 13]),
            convert_block_budgets: false,
        };

        let census_schema =
            load_schema(resource_path(CENSUS_SCHEMA), &rdp13, Some(1)).expect("Loading schema failed");

        // loads requests and converts them to internal format
        let census_requests = load_requests(
            resource_path(CENSUS_REQUESTS),
            &census_schema,
        )
        .expect("Loading requests failed");

        println!("Number of requests: {}", census_requests.len());

        let n_history_requests = (census_requests.len() as f64 * history_frac) as usize;
        let request_history: Vec<Request> = census_requests
            .iter()
            .filter(|(rid, _request)| {
                (RequestId(0) <= **rid) && (**rid < RequestId(n_history_requests))
            })
            .map(|(_rid, request)| request.clone())
            .collect(); //[..n_history_requests];
        let request_batch: Vec<Request> = census_requests
            .iter()
            .filter(|(rid, _request)| **rid >= RequestId(n_history_requests))
            .map(|(_rid, request)| request.clone())
            .collect();

        println!("Percentage of requests part of history: {}", history_frac);
        for eps in epsilons {
            for delta in deltas {
                println!("Params: Epsilon: {} Delta: {}", eps, delta);

                let times = test_algorithm_equality(
                    &census_schema,
                    &request_history,
                    &request_batch,
                    EpsDeltaDp { eps, delta },
                );
                durations.0 += times.0;
                durations.1 += times.1;
            }
        }
        println!(
            "Average durations: Algo hash: {} msec, algo narray: {} msec",
            durations.0 / ((epsilons.len() * deltas.len()) as u128),
            durations.1 / ((epsilons.len() * deltas.len()) as u128)
        )
    }

    fn test_algorithm_equality(
        census_schema: &Schema,
        request_history: &[Request],
        request_batch: &[Request],
        initial_eps_delta_budget: AccountingType,
    ) -> (u128, u128) {
        let initial_budget = &initial_eps_delta_budget.adp_to_rdp_budget(&A13([
            1.5, 1.75, 2., 2.5, 3., 4., 5., 6., 8., 16., 32., 64., 1000000.,
        ]));

        let block = Block {
            id: BlockId(0),
            request_history: request_history.iter().map(|r| r.request_id).collect(),
            default_unlocked_budget: Some(initial_budget.clone()),
            default_total_budget: Some(initial_budget.clone()),
            budget_by_section: Vec::new(),
            created: RoundId(0),
            privacy_unit: PrivacyUnit::User,
            privacy_unit_selection: None,
            retired: None,
        };

        //let history_valid_algo_hash =
        //    test_request_history_algo_hash(request_history, census_schema, initial_budget);
        let _history_valid_algo_narray =
            test_request_history_algo_narray(request_history, census_schema, &block);

        //assert_eq!(
        //    history_valid_algo_hash, history_valid_algo_narray,
        //    "Algorithms returned different results whether history is valid"
        //);
//
        //println!("History Valid: {}", history_valid_algo_hash);

        //println!("running algo_hashmap...");
        //let algo_hash_start = Instant::now();
        //let algo_hash = HashmapSegmentor::new(request_batch.iter().collect(), census_schema);
//
        //let problem_formulation_algo_hash: BlockConstraints<OptimalBudget> =
        //    algo_hash.compute_block_constraints(request_history.iter().collect(), initial_budget);
        //let algo_hash_duration = algo_hash_start.elapsed().as_millis();
        //println!("Algo_hash duration: {} msec", algo_hash_duration);
        let algo_hash_duration = 0;

        println!("running algo_narray...");
        let algo_narray_start = Instant::now();
        let algo_narray = NArraySegmentation::new(request_batch.iter().collect(), census_schema);
        let _problem_formulation_algo_narray: BlockConstraints<OptimalBudget> =
            algo_narray.compute_block_constraints(request_history.iter().collect(), &block, );

        let algo_narray_duration = algo_narray_start.elapsed().as_millis();
        println!("Algo_narray duration: {} msec", algo_narray_duration);
//        let pfs = (
//            problem_formulation_algo_hash,
//            problem_formulation_algo_narray,
//        );
//
//        assert_eq!(
//            pfs.0.acceptable, pfs.1.acceptable,
//            "Accepted requests were not the same : AlgoHash: {:?}, AlgoNarray: {:?}",
//            &pfs.0.acceptable, &pfs.1.acceptable
//        );
//        assert_eq!(
//            pfs.0.rejected, pfs.1.rejected,
//            "Rejected requests were not the same : AlgoHash: {:?}, AlgoNarray: {:?}",
//            &pfs.0.rejected, &pfs.1.rejected
//        );
//        assert_eq!(
//            pfs.0.contested, pfs.1.contested,
//            "Undecided requests were not the same : AlgoHash: {:?}, AlgoNarray: {:?}",
//            &pfs.0.contested, &pfs.1.contested
//        );
//
//        println!(
//            "#Accepted: {}, #Rejected: {}, #Undecided: {}",
//            pfs.0.acceptable.len(),
//            pfs.0.rejected.len(),
//            pfs.0.contested.len()
//        );
//
//        segments_check(
//            pfs.0.contested_segments,
//            pfs.1.contested_segments,
//            "Algo Hash",
//            "Algo Narray",
//        );
//
        (algo_hash_duration, algo_narray_duration)
    }

//    fn test_request_history_algo_hash(
//        request_history: &[Request],
//        schema: &Schema,
//        initial_block_budget: &AccountingType,
//    ) -> bool {
//        let hashmap_segmentor = HashmapSegmentor::new(request_history.iter().collect(), schema);
//        let pf = hashmap_segmentor
//            .compute_block_constraints::<OptimalBudget>(Vec::new(), initial_block_budget);
//        pf.acceptable.len() == request_history.len()
//    }

    fn test_request_history_algo_narray(
        request_history: &[Request],
        schema: &Schema,
        block: &Block,
    ) -> bool {
        let narray_segmentor = NArraySegmentation::new(request_history.iter().collect(), schema);
        let pf = narray_segmentor
            .compute_block_constraints::<OptimalBudget>(Vec::new(), block, );
        pf.acceptable.len() == request_history.len()
    }

//    fn segments_check<M: SegmentBudget + Clone>(
//        segs1: Vec<BlockSegment<M>>,
//        segs2: Vec<BlockSegment<M>>,
//        algo1name: &str,
//        algo2name: &str,
//    ) {
//        assert_eq!(
//            segs1.len(),
//            segs2.len(),
//            "Different number of contested segments"
//        );
//
//        // check that all of the request ids are initialized
//        let mut segment1_map: HashMap<Vec<RequestId>, BlockSegment<M>> = HashMap::new();
//        for mut segment in segs1 {
//            let id_option = segment.request_ids.as_mut();
//            assert!(
//                id_option.is_some(),
//                "Request ids were None for some segment in {}",
//                algo1name
//            );
//            let mut id = id_option.unwrap().clone();
//            assert!(
//                !id.is_empty(),
//                "Some segment in {} had no associated request ids",
//                algo1name
//            );
//            id.sort_unstable();
//            let inserted = segment1_map.insert(id.clone(), segment);
//            assert!(
//                inserted.is_none(),
//                "Duplicate request id in {}: {:?}",
//                algo1name,
//                &id
//            );
//        }
//
//        for mut segment in segs2 {
//            let id_option = segment.request_ids.as_mut();
//            assert!(
//                id_option.is_some(),
//                "Request ids were none for some segment in {}",
//                algo2name
//            );
//            let mut id = id_option.unwrap().clone();
//            assert!(
//                !id.is_empty(),
//                "Some segment in {} had no associated request ids",
//                algo2name
//            );
//            id.sort_unstable();
//            let removed = segment1_map.remove(&id);
//            assert!(
//                removed.is_some(),
//                "Segment with request ids {:?} of {} not found in {}",
//                &id,
//                algo2name,
//                algo1name
//            );
//            assert!(removed
//                .unwrap()
//                .remaining_budget
//                .approx_eq(&segment.remaining_budget, F64Margin::default()));
//        }
//
//        assert!(
//            // should actually never trigger this, as segs1.len() == segs2.len()
//            segment1_map.is_empty(),
//            "Some segments of {} not present in {}",
//            algo1name,
//            algo2name
//        );
//    }
}

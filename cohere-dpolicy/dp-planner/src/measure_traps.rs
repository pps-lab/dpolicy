use clap::Parser;
use dp_planner_lib::{
    block::{self, Block, BlockId},
    composition::block_composition_pa::{
        algo_narray::{AdpCost, NArraySegmentation},
        Segmentation,
    },
    config::BudgetTotal,
    dprivacy::{
        privacy_unit::{MyIntervalSet, PrivacyUnit},
        rdp_alphas_accounting::PubRdpAccounting,
        AccountingType,
    },
    request::{self, Request, RequestId},
    schema,
    simulation::RoundId,
};
use gcollections::ops::set::Overlap;
use gcollections::ops::{Bounded, IsSingleton};
use interval::IntervalSet;
use itertools::Itertools;
use rayon::prelude::*;
use serde::{Deserialize, Deserializer, Serialize};
use std::str::FromStr;
use std::{collections::HashSet, io::Write};
use std::{
    collections::{BTreeMap, HashMap},
    fs::File,
    path::PathBuf,
};
use interval::ops::Width;
use regex::Regex;
use crate::AttributeDim::Attribute;

#[derive(Parser, Debug, Clone)]
#[clap(author, version, about, long_about = None)]
pub struct Cli {
    #[clap(long, parse(from_os_str), value_name = "DIR")]
    pub dir: PathBuf,

    #[clap(long, parse(from_os_str), value_name = "DIR")]
    pub output_dir: Option<PathBuf>,

    #[clap(long, default_value = "trap_analysis.json")]
    pub output: String,

    #[clap(long, default_value = "all_blocks.json")]
    pub blocks: String,

    #[clap(long, default_value = "all_requests.json")]
    pub requests: String,

    #[clap(long, default_value = "schema.json")]
    pub schema: String,

    #[clap(long, default_value = "request_log.csv")]
    pub request_log: String,

    #[clap(long, arg_enum)]
    pub privacy_unit: PrivacyUnit,

    #[clap(
        long,
        value_name = "TIME_PRIVACY_UNIT",
        help = "an array of TimePrivacyUnit"
    )]
    // --time-privacy-unit {"unit": "UserMonth", "min": 7, "max": 15}   --time-privacy-unit {"unit": "UserDay", "min": 7, "max": 15}
    pub time_privacy_unit: Vec<TimePrivacyUnit>,

    /// converts epsilon, delta approximate differential privacy budget to renyi differential privacy
    /// budget, using the given alpha values. Only 1, 2, 3, 4, 5, 7, 10, 13, 14 or 15 values are supported.
    /// See [AdpAccounting::adp_to_rdp_budget] for more details
    #[clap(
        long,
        min_values(1),
        max_values(15),
        value_delimiter = ' ',
        default_value = "1.5 1.75 2.0 2.5 3.0 4.0 5.0 6.0 8.0 16.0 32.0 64.0 1.0e6 1.0e10"
    )]
    pub alphas: Vec<f64>,

    /// If set to true, converts unlocked budgets of blocks from adp to rdp, same as the budget passed
    /// by the command line. See [AdpAccounting::adp_to_rdp_budget] for more details
    #[clap(long, default_value = "true")]
    pub convert_block_budgets: bool,

    // The delta value used to compute the max privacy costs
    #[clap(long)]
    pub delta: f64,

    #[clap(long, arg_enum, default_value("accepted"))]
    pub request_mode: RequestMode,
}

/// Which solver should be used to (approximately) solve Knapsack.
/// See [allocation::efficiency_based::knapsack::KPApproxSolver] for more details.
#[derive(clap::ArgEnum, Serialize, Deserialize, Debug, Clone, Copy)]
pub enum RequestMode {
    Accepted,
    All, // also account for rejected requests
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TimePrivacyUnit {
    pub unit: PrivacyUnit,
    //  [0, min-1], [min], [min + 1], ... , [max], [max+1, +inf]
    pub min: u64,
    pub max: u64,
}

impl FromStr for TimePrivacyUnit {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {

        // Because the command line is very annoying with JSON quotes, we also accept non-quoted strings
        // Step 1: Add quotes around unquoted keys
        // Step 1: Match unquoted keys and replace them with quoted keys
        let re_keys = Regex::new(r"(\b\w+\b)\s*:").unwrap();
        let result = re_keys.replace_all(s, r#""$1":"#);

        // Step 2: Match unquoted string values (only alphabetic values) and quote them
        let re_values = Regex::new(r#":\s*([a-zA-Z]+)(\s*[},])"#).unwrap();
        let result = re_values.replace_all(&result, r#": "$1"$2"#);

        serde_json::from_str(&result).map_err(|e| format!("error parsing time privacy unit: {}", e))
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, PartialOrd, Eq, Ord)]
pub enum AttributeDim {
    All,
    Attribute(String),
    Category(String),
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct Stat {
    attribute_dim: AttributeDim,
    relaxation: String,
    privacy_unit: PrivacyUnit,
    privacy_unit_selection: Option<MyIntervalSet>,
    cost: Option<AccountingType>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, PartialOrd, Eq, Ord)]
struct MinMax(u64, u64);

fn main() {
    let args: Cli = Cli::parse();

    let blocks_file = args.dir.clone().join(args.blocks);
    let requests_file = args.dir.clone().join(args.requests);
    let schema_file = args.dir.clone().join(args.schema);
    let request_log_file = args.dir.clone().join(args.request_log);

    let budget_total = BudgetTotal {
        privacy_units: Some(vec![args.privacy_unit.clone()]),
        alphas: Some(args.alphas.clone()),
        convert_block_budgets: args.convert_block_budgets,
    };

    let alphas = budget_total.alphas().unwrap();

    // load and init schema
    let schema: schema::Schema =
        schema::load_schema_pa(schema_file, &budget_total, None).expect("loading schema pa failed");

    println!("Loading candidate requests...");
    // loads candidate requests and converts them to internal format
    let (candidate_requests, attribute_lookup, category_lookup, relaxation_lookup) =
        request::load_requests_pa(requests_file, &schema);

    println!("Loaded {} candidate requests", candidate_requests.len());

    let request_history = HashMap::new();
    println!("Loading blocks...");
    let blocks: HashMap<BlockId, Block> =
        block::load_blocks(blocks_file, &request_history, &schema, &budget_total)
            .expect("loading of blocks failed");
    println!("Loaded {} blocks", blocks.len());

    let requests_per_block: BTreeMap<BlockId, HashSet<RequestId>> = load_requests_per_block(
        request_log_file,
        args.privacy_unit.clone(),
        args.request_mode.clone(),
        &blocks,
    );

    let bounds_by_privacy_unit = process_time_privacy_unit(&args.time_privacy_unit, &blocks);

    let stats = get_statistics(
        &args.privacy_unit,
        relaxation_lookup,
        attribute_lookup,
        category_lookup,
        bounds_by_privacy_unit,
        &candidate_requests,
    );
    let stats_requests: Vec<(usize, HashSet<RequestId>)> = stats
        .iter()
        .enumerate()
        .map(|(i, (_, request_ids))| (i, request_ids.clone()))
        .collect();
    let stats: Vec<Stat> = stats.into_iter().map(|(stat, _)| stat).collect();

    let mut max_across_blocks = HashMap::new();

    for (idx, (block_id, request_ids)) in requests_per_block.iter().enumerate() {
        println!(
            "Block: {:?} - n_accepted_requests: {:?}",
            block_id,
            request_ids.len()
        );

        let window_size = schema.block_sliding_window_size;
        if idx < window_size - 1 || idx > requests_per_block.len() - window_size {
            println!("  -> skipping this block: {:?} (first or last window_size={:?}-1 blocks cannot have the max privacy usage", block_id, window_size);
            continue;
        }

        let block_allocated_requests: Vec<&Request> = request_ids
            .iter()
            .map(|rid| candidate_requests.get(rid).unwrap())
            .collect();

        let segmentation = NArraySegmentation::new(block_allocated_requests.clone(), &schema);
        let segments = segmentation.compute_idx_per_segment();

        // in parallel compute the max adp cost for each stat
        //   by taking the requests allocated on a block and filtering them depending on whether the request is relevant for the stat.
        let stats_epsilon: Vec<(usize, f64)> = stats_requests
            .par_iter()
            .map(|(i, matching_request_ids)| {
                // filter the trap relevant requests
                let relevant_requests: Vec<&Request> = block_allocated_requests
                    .iter()
                    .filter(|request| matching_request_ids.contains(&request.request_id))
                    .map(|r| *r)
                    .collect();

                let n_relevant_requests = relevant_requests.len();

                let max_adp_by_block = segmentation.max_adp_cost(
                    &segments,
                    relevant_requests,
                    args.privacy_unit.clone(),
                    &alphas,
                    args.delta,
                );

                let stat = stats.get(*i).unwrap();

                println!(
                    "  BlockId={:?}  Stat={:?}   n_relevant_requests={:?} max_adp_cost={:?} ",
                    block_id, stat, n_relevant_requests, max_adp_by_block
                );

                let epsilon = match max_adp_by_block {
                    AccountingType::EpsDeltaDp { eps, .. } => eps,
                    _ => panic!("Expected EpsilonDp"),
                };

                (*i, epsilon)
            })
            .collect();

        // update the max epsilon across blocks (seen so far)
        for (i, epsilon) in stats_epsilon.into_iter() {
            max_across_blocks
                .entry(i)
                .and_modify(|x| *x = f64::max(epsilon, *x))
                .or_insert(epsilon);
        }
    }

    let stats: Vec<Stat> = stats
        .iter()
        .enumerate()
        .map(|(i, stat)| {
            let max_epsilon = max_across_blocks.get(&i).unwrap().clone();

            let mut stat = stat.clone();

            stat.cost = Some(AccountingType::EpsDeltaDp {
                eps: max_epsilon,
                delta: args.delta,
            });

            stat
        })
        .sorted_by(|a, b| {
            let order = a
                .privacy_unit
                .cmp(&b.privacy_unit)
                .then_with(|| a.relaxation.cmp(&b.relaxation))
                .then_with(|| a.attribute_dim.cmp(&b.attribute_dim));
            order
        })
        .collect();

    println!("================================================");
    stats.iter().for_each(|stat| {
        println!("{:?}", stat);
    });
    println!("================================================");

    // output the results to a json file
    let output = Output {
        privacy_unit: args.privacy_unit.clone(),
        alphas: alphas.to_vec(),
        request_mode: args.request_mode.clone(),
        stats,
    };
    let json_string = serde_json::to_string_pretty(&output).unwrap();
    let output_file = args.output_dir.unwrap_or(args.dir).join(args.output);
    let mut file = File::create(output_file).expect("Unable to create file");
    file.write_all(json_string.as_bytes())
        .expect("Unable to write data");
}

fn process_time_privacy_unit(time_privacy_units: &Vec<TimePrivacyUnit>, blocks: &HashMap<BlockId, Block>) -> HashMap<PrivacyUnit, MinMax> {
    let bounds_by_privacy_unit: HashMap<PrivacyUnit, MinMax> = time_privacy_units
        .iter()
        .map(|tpu| (tpu.unit.clone(), MinMax(tpu.min, tpu.max)))
        .collect();

    // Computing the actual bounds from the block only works from DPolicy (and not Cohere) -> hence we have the bounds provided in the CLI
    //  In the case of DPolicy, we actually compare whether they are set correctly.
    let mut actual_bounds_by_privacy_unit: HashMap<PrivacyUnit, MinMax> = HashMap::new();
    blocks.iter().for_each(|(_, block)| {
        if let Some(interval_set) = &block.privacy_unit_selection {
            if interval_set.is_singleton() {
                let value: u64 = interval_set.lower();
                let upper: u64 = interval_set.upper();
                assert_eq!(value, upper, "Expected a singleton interval");

                actual_bounds_by_privacy_unit
                    .entry(block.privacy_unit.clone())
                    .and_modify(|x| {
                        x.0 = u64::min(value, x.0);
                        x.1 = u64::max(value, x.1);
                    })
                    .or_insert(MinMax(value, value));
            }
        }
    });
    assert!(actual_bounds_by_privacy_unit.len() == 0 || actual_bounds_by_privacy_unit == bounds_by_privacy_unit, "Expected the actual bounds to match the bounds provided in the CLI");
    bounds_by_privacy_unit
}

fn load_requests_per_block(
    request_log_file: PathBuf,
    privacy_unit: PrivacyUnit,
    request_mode: RequestMode,
    blocks: &HashMap<BlockId, Block>,
) -> BTreeMap<BlockId, HashSet<RequestId>> {
    let file = File::open(request_log_file).expect("Could not open file");
    let mut rdr = csv::Reader::from_reader(file);

    let requests_logs: Vec<MyRequestLogRow> =
        rdr.deserialize().map(|record| record.unwrap()).collect();
    println!("Loaded {} request logs", requests_logs.len());

    // for each decision round figure which blocks were assigned
    let mut blocks_per_decision_round = BTreeMap::new();
    match request_mode {
        RequestMode::Accepted => {}
        RequestMode::All => {
            for log in requests_logs.iter() {
                match log.decision.as_str() {
                    "Accepted" => {
                        let blocks = &log.assigned_blocks.0;
                        match blocks_per_decision_round.entry(log.decision_round) {
                            std::collections::btree_map::Entry::Vacant(e) => {
                                e.insert(blocks.clone());
                            }
                            std::collections::btree_map::Entry::Occupied(e) => {
                                assert_eq!(
                                    e.get(),
                                    blocks,
                                    "Blocks assigned in the same decision round are not the same"
                                );
                            }
                        }
                    }
                    "Rejected" => {}
                    _ => panic!("Unknown decision: {}", log.decision),
                }
            }
        }
    }
    // TODO: WE COULD ALSO ADD ANOTHER WAY TO LOAD "ALL REQUESTS" TO ANALYZE THE CUMULATIVE COST OF A WORKLOAD WITHOUT RUNNING AN ALLOCATION
    //         -> The problem is that from the wo
    let mut data = BTreeMap::new();
    for log in requests_logs.iter() {
        match log.decision.as_str() {
            "Accepted" => {
                for block_id in log.assigned_blocks.0.iter() {
                    // filter out the blocks that are not from the main privacy unit
                    let block = blocks.get(block_id).unwrap();
                    if block.privacy_unit == privacy_unit {
                        data.entry(block_id.clone())
                            .or_insert_with(HashSet::new)
                            .insert(log.request_id);
                    } else {
                        println!(
                            "  -> skipping this block: {:?} (privacy unit does not match)",
                            block_id
                        );
                    }
                }
            }
            "Rejected" => {
                match request_mode {
                    RequestMode::Accepted => {}
                    RequestMode::All => {
                        // we can also consider rejected requests to obtain the max privacy cost of the workload
                        assert!(log.decision_is_final, "We assume only final decisions here");
                        let block_ids = blocks_per_decision_round
                            .get(&log.decision_round)
                            .expect("decision round not present");
                        for block_id in block_ids.iter() {
                            // filter out the blocks that are not from the main privacy unit
                            let block = blocks.get(block_id).unwrap();
                            if block.privacy_unit == privacy_unit {
                                data.entry(block_id.clone())
                                    .or_insert_with(HashSet::new)
                                    .insert(log.request_id);
                            } else {
                                println!(
                                    "  -> skipping this block: {:?} (privacy unit does not match)",
                                    block_id
                                );
                            }
                        }
                    }
                }
            }
            _ => panic!("Unknown decision: {}", log.decision),
        }
    }
    data
}

fn privacy_unit_selection_stats(
    attribute_dim: AttributeDim,
    relaxation: String,
    privacy_unit: PrivacyUnit,
    min_singleton: u64,
    max_singleton: u64,
) -> Vec<Stat> {
    let mut stats = Vec::new();

    let stat = Stat {
        attribute_dim,
        relaxation,
        privacy_unit,
        privacy_unit_selection: None,
        cost: None,
    };

    if min_singleton > 0 {
        let mut x = stat.clone();
        x.privacy_unit_selection = Some(MyIntervalSet::new(0, min_singleton - 1));
        stats.push(x);
    }
    for i in min_singleton..=max_singleton {
        let mut x = stat.clone();
        x.privacy_unit_selection = Some(MyIntervalSet::new(i, i));
        stats.push(x);
    }
    let max_value = <u64 as Width>::max_value();
    if max_singleton < max_value {
        let mut x = stat.clone();
        x.privacy_unit_selection = Some(MyIntervalSet::new(max_singleton + 1, max_value));
        stats.push(x);
    }
    stats
}

fn get_statistics(
    main_privacy_unit: &PrivacyUnit,
    relaxation_lookup: BTreeMap<String, HashSet<RequestId>>,
    attribute_lookup: BTreeMap<String, HashSet<RequestId>>,
    category_lookup: BTreeMap<String, HashSet<RequestId>>,
    bounds_by_privacy_unit: HashMap<PrivacyUnit, MinMax>,
    requests: &HashMap<RequestId, Request>,
) -> Vec<(Stat, HashSet<RequestId>)> {
    let mut stats: Vec<(Stat, HashSet<RequestId>)> = Vec::new();

    for (relax, relax_request_ids) in relaxation_lookup.iter() {
        // independent of attributes (any attribute)
        let all = Stat {
            attribute_dim: AttributeDim::All,
            relaxation: relax.clone(),
            privacy_unit: main_privacy_unit.clone(),
            privacy_unit_selection: None,
            cost: None,
        };
        stats.push((all, relax_request_ids.clone()));

        // per-attribute
        for (attribute, attribute_request_ids) in attribute_lookup.iter() {
            let per_attr = Stat {
                attribute_dim: AttributeDim::Attribute(attribute.clone()),
                relaxation: relax.clone(),
                privacy_unit: main_privacy_unit.clone(),
                privacy_unit_selection: None,
                cost: None,
            };

            let request_ids = relax_request_ids
                .intersection(attribute_request_ids)
                .copied()
                .collect();

            stats.push((per_attr, request_ids));
        }

        // per-category
        for (category, category_request_ids) in category_lookup.iter() {
            let per_cat = Stat {
                attribute_dim: AttributeDim::Category(category.clone()),
                relaxation: relax.clone(),
                privacy_unit: main_privacy_unit.clone(),
                privacy_unit_selection: None,
                cost: None,
            };
            let request_ids = relax_request_ids
                .intersection(category_request_ids)
                .copied()
                .collect();
            stats.push((per_cat, request_ids));
        }

        let attribute_dim = AttributeDim::All;
        let mut privacy_unit_stats: Vec<(Stat, Option<&HashSet<RequestId>>)> = bounds_by_privacy_unit
            .iter()
            .flat_map(|(unit, minmax)| {
                let stats = privacy_unit_selection_stats(
                    attribute_dim.clone(),
                    relax.clone(),
                    unit.clone(),
                    minmax.0,
                    minmax.1,
                );
                stats.iter().map(|stat| (stat.clone(), None)).collect::<Vec<(Stat, Option<&HashSet<RequestId>>)>>()
            })
            .collect();

        if attribute_lookup.len() < 10 {
            let extra_privacy_unit_stats: Vec<(Stat, Option<&HashSet<RequestId>>)> = bounds_by_privacy_unit
                .iter()
                .flat_map(|(unit, minmax)| {
                    attribute_lookup.iter().flat_map(|(attribute, attribute_request_ids)| {
                        let stats = privacy_unit_selection_stats(
                            Attribute(attribute.clone()),
                            relax.clone(),
                            unit.clone(),
                            minmax.0,
                            minmax.1,
                        );
                        stats.iter().map(|stat| (stat.clone(), Some(attribute_request_ids))).collect::<Vec<_>>()
                    })
                })
                .collect();
            privacy_unit_stats.extend(extra_privacy_unit_stats);
        } else {
            println!("Skipping per-attribute privacy unit stats because there are too many attributes");
        }

        for stat in privacy_unit_stats {
            let stat_selection: &IntervalSet<u64> = stat
                .0.privacy_unit_selection
                .as_ref()
                .expect("Expected a privacy unit selection");

            let mut request_ids: HashSet<RequestId> = relax_request_ids
                .iter()
                .filter(|rid| {
                    // compute request ids from requests that overlap with the interval defined by the stat
                    let request = requests.get(rid).expect("Request not found");
                    let request_selection: &IntervalSet<u64> = request
                        .privacy_unit_selection
                        .get(&stat.0.privacy_unit)
                        .expect("Privacy unit missing");
                    request_selection.overlap(stat_selection)
                })
                .copied()
                .collect();

            if let Some(attribute_request_ids) = stat.1 {
                request_ids = request_ids
                    .intersection(attribute_request_ids)
                    .copied()
                    .collect();
            }

            stats.push((stat.0, request_ids));
        }
    }
    stats
}

#[derive(Serialize, Deserialize, Debug)]
struct Output {
    privacy_unit: PrivacyUnit,
    alphas: Vec<f64>,
    request_mode: RequestMode,
    stats: Vec<Stat>,
}

/// Contains per-request information, written to req-log-output
#[derive(Deserialize)]
pub struct MyRequestLogRow {
    decision_round: RoundId,
    decision: String,
    request_id: RequestId,
    request_cost: String,
    num_blocks: usize,
    profit: u64,
    assigned_blocks: VecBlockId,
    decision_is_final: bool,
    num_virtual_blocks: usize,
}

struct VecBlockId(Vec<BlockId>);

impl<'de> Deserialize<'de> for VecBlockId {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        // println!("Deserializing VecBlockId: {}", s);
        let v: Vec<BlockId> = serde_json::from_str(&s).expect("Could not parse VecBlockId");
        Ok(VecBlockId(v))
    }
}

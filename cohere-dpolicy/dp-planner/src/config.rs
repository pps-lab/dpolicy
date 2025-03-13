use crate::dprivacy::privacy_unit::PrivacyUnit;
use crate::dprivacy::rdp_alphas_accounting::{PubRdpAccounting, RdpAlphas};
use clap::{Parser, Subcommand};
use std::ffi::OsStr;
use std::path::PathBuf;

#[derive(Parser, Debug, Clone)]
#[clap(author, version, about, long_about = None)]

pub struct Cli {
    #[clap(subcommand)]
    pub mode: Mode,

    #[clap(flatten)]
    pub input: Input,

    #[clap(flatten)]
    pub output_config: OutputConfig,
}

#[derive(Subcommand, Debug, Clone)]
pub enum Mode {
    Simulate {
        #[clap(subcommand)]
        allocation: AllocationConfig,

        /// If keep_rejected_requests is set, this option limits long requests are kept. Set to a
        /// number higher than the number of rounds to keep all requests.
        #[clap(long, short, default_value("1"))]
        timeout_rounds: usize,

        /// If set, this option limits how many requests can be processed. Useful to generate a
        /// history with some remaining requests.
        #[clap(long, short)]
        max_requests: Option<usize>,
    },
    Round {
        #[clap(subcommand)]
        allocation: AllocationConfig,

        /// Round number
        i: usize,
    },
}

#[derive(Subcommand, Debug, Clone)]
pub enum AllocationConfig {
    /// Greedy allocation algorithm (prioritizes lower request id).
    //Greedy {
    //    #[clap(subcommand)]
    //    composition: CompositionConfig,
    //},

    /// Solve a profit optimization problem formulated as an integer linear program (ilp).
    //Ilp {
    //    #[clap(subcommand)]
    //    composition: CompositionConfig,
    //},

    /// Dominant Private Block Fairness allocation algorithm from the Privacy Budget Scheduling paper.
    //Dpf {
    //    /// The seed used in deciding which blocks are desired by each request
    //    #[clap(long, default_value("42"))]
    //    block_selector_seed: u64,
//
    //    /// If set, the weighted dpf algorithm is used, which is a modification of the original dpf
    //    /// as described in "Packing Privacy Budget Efficiently" by Tholoniat et al
    //    #[clap(long)]
    //    weighted_dpf: bool,
//
    //    /// If set, the dpf (and weighted dpf) consider the remaining budget of the selected blocks to determine the dominant share.
    //    /// In the original Luo et al 2021 paper, the share is determined by the global budget.
    //    /// In "Packing Privacy Budget Efficiently" by Tholoniat et al 2022, the share is determined by the remaining budget of the selected blocks.
    //    #[clap(long)]
    //    dominant_share_by_remaining_budget: bool,
//
    //    #[clap(subcommand)]
    //    composition: CompositionConfig,
    //},

    /// Any efficiency-based allocation algorithms (currently only Dpk) except dpf, for which a
    /// separate, optimized implementation exists.
    EfficiencyBased {
        /// The type of efficiency-based algorithm to use
        #[clap(subcommand)]
        algo_type: EfficiencyBasedAlgo,

        /// The seed used in deciding which blocks are desired by each request
        #[clap(long, default_value("42"))]
        block_selector_seed: u64,
    },
}

/// The type of efficiency-based algorithm to use.
///
/// Add any new efficiency-based algos here, and fix any compiler errors with match clauses to allow
/// access to the new algo via the CLI.
#[derive(Subcommand, Debug, Clone)]
pub enum EfficiencyBasedAlgo {
    /// use Dpk
    Dpk {
        /// determines how close to the optimal solution the knapsack solver should be. Lower values
        /// result in better approximations, but also in longer runtimes. Should be between 0 and 0.75
        /// (ends not included).
        #[clap(long, default_value("0.05"))]
        eta: f64,

        /// Which solver should be used to (approximately) solve Knapsack.
        #[clap(long, arg_enum, default_value("fptas"))]
        kp_solver: KPSolverType,

        /// How many parallel instances of
        /// [kp_solver](enum.EfficiencyBasedAlgo.html#variant.Dpk.field.kp_solver) should run in
        /// parallel at most at any time
        #[clap(long)]
        num_threads: Option<usize>,

        #[clap(subcommand)]
        composition: CompositionConfig,
    },
}

impl EfficiencyBasedAlgo {
    pub fn get_composition(&self) -> &CompositionConfig {
        match self {
            Self::Dpk { composition, .. } => composition,
        }
    }
}

/// Which solver should be used to (approximately) solve Knapsack.
/// See [allocation::efficiency_based::knapsack::KPApproxSolver] for more details.
#[derive(clap::ArgEnum, Debug, Clone, Copy)]
pub enum KPSolverType {
    FPTAS,
    Gurobi,
}

#[derive(clap::ArgEnum, Debug, Clone)]
pub enum BudgetType {
    /// use OptimalBudget
    OptimalBudget,
    /// use RdpMinBudget
    RdpMinBudget,
}

#[derive(Subcommand, Debug, Clone)]
pub enum CompositionConfig {
    /// Block composition with Partitioning Attributes.
    BlockCompositionPa {
        #[clap(subcommand)]
        budget: Budget,

        /// The segmentation algo to split the request batch into segments and compute the remaining budget.
        #[clap(short, long, arg_enum, default_value("narray"))]
        algo: SegmentationAlgo,

        /// For how many blocks in parallel we compute the budget constraints.
        #[clap(long)]
        num_threads: Option<usize>,

        #[clap(short, long, arg_enum, default_value("optimal-budget"))]
        budget_type: BudgetType,
    },
    /// Regular block composition (without partitioning attributes)
    BlockComposition {
        #[clap(subcommand)]
        budget: Budget,

        #[clap(short, long, arg_enum, default_value("optimal-budget"))]
        budget_type: BudgetType,
    },
}

#[derive(clap::ArgEnum, Debug, Clone, Copy)]
pub enum SegmentationAlgo {
    Narray,
//    Hashmap,
}

#[derive(Subcommand, Debug, Clone)]
pub enum Budget {
    /// The complete budget is already unlocked in the first round.
    FixBudget {
        #[clap(flatten)]
        budget: BudgetTotal,
    },

    /// The budget is gradually unlocked over time (i.e., requests in the first round cannot
    /// consume the complete budget). Note that for both greedy and dpf, selecting
    /// this option will throw an error if the
    /// [batch_size](enum.Mode.html#variant.Simulate.field.batch_size) is > 1.
    /// Therefore, for both greedy and dpf, the budget unlocked per round is total_budget / n_steps,
    /// regardless of if the trigger is set to round or request.
    /// For ilp, it works as expected: in each round, the amount of budget unlocked is either
    /// total_budget / n_steps if the trigger is round, or batch_size / n_steps if the trigger is
    /// request.
    UnlockingBudget {
        /// The trigger of a budget unlocking step.
        #[clap(short, long, arg_enum)]
        trigger: UnlockingBudgetTrigger,

        // would require additional state on block
        //#[clap(short, long)]
        //every: usize,
        /// The total number of unlocking steps.
        #[clap(short, long)]
        n_steps: usize,

        /// The slack \in [0, 1] unlocks slightly more budget in the first n_steps/2 unlocking steps:  (1 + slack) * budget/n_steps
        /// and then (1 - slack) * budget/n_steps in the 2nd part of the unlocking steps.
        /// Currently, slack can only be used if the trigger is set to round (slack default = 0.0).
        #[clap(short, long)]
        slack: Option<f64>,

        /// The total amount of budget available over all unlocking steps.
        #[clap(flatten)]
        budget: BudgetTotal,
    },
}

#[derive(clap::Args, Debug, Clone)]
pub struct BudgetTotal {

    #[clap(long, min_values(1), arg_enum)]
    pub privacy_units: Option<Vec<PrivacyUnit>>,

    /// converts epsilon, delta approximate differential privacy budget to renyi differential privacy
    /// budget, using the given alpha values. Only 1, 2, 3, 4, 5, 7, 10, 13, 14 or 15 values are supported.
    /// See [AdpAccounting::adp_to_rdp_budget] for more details
    #[clap(long, min_values(1), max_values(15))]
    pub alphas: Option<Vec<f64>>,


    /// If set to true, converts unlocked budgets of blocks from adp to rdp, same as the budget passed
    /// by the command line. See [AdpAccounting::adp_to_rdp_budget] for more details
    #[clap(long, requires("alphas"))]
    pub convert_block_budgets: bool,
}

impl Budget {

    /// Returns whether or not the budget is unlocking or not
    pub fn unlocking_budget(&self) -> bool {
        match self {
            Budget::FixBudget { .. } => false,
            Budget::UnlockingBudget { .. } => true,
        }
    }
}

impl BudgetTotal {

    pub fn new() -> Self {
        BudgetTotal{
            privacy_units: Some(vec![PrivacyUnit::User]),
            alphas: None,
            convert_block_budgets: false,
        }
    }

    /// Returns the alphas passed to the Cli if they were passed and block budgets should be
    /// converted
    pub fn block_conversion_alphas(&self) -> Option<RdpAlphas> {
        if let BudgetTotal {
            convert_block_budgets: true,
            ..
        } = self
        {
            self.alphas()
        } else {
            None
        }
    }


    /// Returns the alphas passed to the Cli if they were passed
    pub fn alphas(&self) -> Option<RdpAlphas> {
        if let BudgetTotal {
            alphas: Some(alpha_vals),
            ..
        } = self
        {
            Some(RdpAlphas::from_vec(
                alpha_vals.clone()).expect(
                "Supplied an unsupported number of alpha values (supported: 1, 2, 3, 4, 5, 7, 10, 13, 14, 15)"
            )
            )
        } else {
            None
        }
    }
}

#[derive(clap::ArgEnum, Debug, Clone)]
pub enum UnlockingBudgetTrigger {
    Round,
//    Request,
}

#[derive(clap::Args, Debug, Clone)]
pub struct Input {
    /// Schema file of partitioning attributes
    #[clap(short = 'S', long, parse(from_os_str), value_name = "FILE")]
    pub schema: PathBuf,

    /// Existing blocks with request history
    #[clap(short = 'B', long, parse(from_os_str), value_name = "FILE")]
    pub blocks: PathBuf,

    /// Candidate requests for allocation
    #[clap(short = 'R', long, parse(from_os_str), value_name = "FILE")]
    pub requests: PathBuf,

//    /// Config for request adapter to set request cost, block demand, and profit.
//    #[clap(flatten)]
//    pub request_adapter_config: RequestAdapterConfig,

    /// Previously accepted requests
    #[clap(short = 'H', long, parse(from_os_str), value_name = "FILE")]
    pub history: Option<PathBuf>,
}

//#[derive(clap::Args, Debug, Clone)]
//pub struct RequestAdapterConfig {
//    /// Sets the file which contains the request adapter (if not set, empty adapter is used)
//    #[clap(short = 'A', long, parse(from_os_str), value_name = "FILE")]
//    pub request_adapter: Option<PathBuf>,
//
//    /// Sets the seed which is used for the request adapter
//    #[clap(long, requires("request-adapter"))]
//    pub request_adapter_seed: Option<u128>,
//}

#[derive(clap::Args, Debug, Clone)]
pub struct OutputConfig {
    /// Sets the path for the log file, containing information about each request
    #[clap(
        long,
        parse(from_os_str),
        value_name = "FILE",
        default_value("./results/requests.csv")
    )]
    pub req_log_output: PathBuf,

    /// Sets the path for the log file, containing information about each round
    #[clap(
        long,
        parse(from_os_str),
        value_name = "FILE",
        default_value("./results/rounds.csv")
    )]
    pub round_log_output: PathBuf,

    /// Sets the path for the log file, containing information about the round runtime
    #[clap(
        long,
        parse(from_os_str),
        value_name = "FILE",
        default_value("./results/runtime.csv")
    )]
    pub runtime_log_output: PathBuf,

    /// Whether or not the remaining budget is logged as part of the round log. Warning: This can
    /// be expensive, especially with a small batch size.
    #[clap(long)]
    pub log_remaining_budget: bool,

    /// Whether or not nonfinal rejections are logged
    #[clap(long)]
    pub log_nonfinal_rejections: bool,

    /// Sets the path to the stats file, containing summary metrics of the current run
    #[clap(
        long,
        parse(from_os_str),
        value_name = "FILE",
        default_value("./results/stats.json")
    )]
    pub stats_output: PathBuf,

    /// Optionally define a directory where the generated history and blocks is saved. The files
    /// will have paths history_output_directory/block_history.json,
    /// history_output_directory/request_history.json and
    /// history_output_directory/remaining_requests.json
    #[clap(long, parse(from_os_str), value_name = "FILE")]
    pub history_output_directory: Option<PathBuf>,
}

// provide top-level access for different parts of the config
impl Cli {
    pub fn total_budget(&self) -> &BudgetTotal {
        match self.composition(){
            CompositionConfig::BlockCompositionPa { budget, ..} => match budget {
                Budget::FixBudget { budget, .. } => budget,
                Budget::UnlockingBudget {  budget, .. } => budget,
            },
            CompositionConfig::BlockComposition { budget, ..} => match budget {
                Budget::FixBudget { budget } => budget,
                Budget::UnlockingBudget {budget, .. } => budget,
            },
        }
    }

    pub fn block_sliding_window_size(&self) -> Option<usize>{

        match self.composition(){
            CompositionConfig::BlockCompositionPa { budget, ..} => match budget {
                Budget::FixBudget { .. } => None,
                Budget::UnlockingBudget {  n_steps, .. } => Some(*n_steps),
            },
            CompositionConfig::BlockComposition { budget, ..} => match budget {
                Budget::FixBudget { .. } => None,
                Budget::UnlockingBudget {n_steps, .. } => Some(*n_steps),
            },
        }
    }


    fn composition(&self) -> &CompositionConfig {
        self.mode.composition()
    }

    pub fn allocation(&self) -> &AllocationConfig {
        self.mode.allocation()
    }

    /// Checks that the combination of selected features is currently supported (where this is not
    /// already handled via [attributes](https://doc.rust-lang.org/reference/attributes.html))
    pub fn check_config(&self) {
        // check that the mode is simulate - round is not currently supported
        match self.mode {
            Mode::Simulate { timeout_rounds, .. } => {
                assert!(
                    0 < timeout_rounds,
                    "Timeout rounds must be strictly positive"
                );
            }
            Mode::Round { .. } => {
                panic!("Round mode is not currently supported, use simulate instead")
            }
        }
    }
}

impl Mode {
    fn composition(&self) -> &CompositionConfig {
        match self {
            Mode::Simulate { allocation, .. } => allocation.composition(),
            Mode::Round { allocation, .. } => allocation.composition(),
        }
    }


    fn allocation(&self) -> &AllocationConfig {
        match self {
            Mode::Simulate { allocation, .. } => allocation,
            Mode::Round { allocation, .. } => allocation,
        }
    }
}

impl EfficiencyBasedAlgo {
    pub(crate) fn composition(&self) -> &CompositionConfig {
        match self {
            EfficiencyBasedAlgo::Dpk { composition, .. } => composition,
        }
    }

    pub fn budget_config(&self) -> &Budget {
        match self {
            EfficiencyBasedAlgo::Dpk { composition, .. } => composition.budget_config(),
        }
    }
}

impl AllocationConfig {

    pub(crate) fn composition(&self) -> &CompositionConfig {
        match self {
            //AllocationConfig::Dpf {
            //    block_selector_seed: _,
            //    composition,
            //    ..
            //} => composition.budget(),
            //AllocationConfig::Greedy { composition } => composition.budget(),
            //AllocationConfig::Ilp { composition } => composition.budget(),
            AllocationConfig::EfficiencyBased { algo_type, .. } => algo_type.composition(),
        }
    }


    pub fn budget_config(&self) -> &Budget {
        match self {
            //AllocationConfig::Dpf {
            //    block_selector_seed: _,
            //    composition,
            //    ..
            //} => composition.budget_config(),
            //AllocationConfig::Greedy { composition } => composition.budget_config(),
            //AllocationConfig::Ilp { composition } => composition.budget_config(),
            AllocationConfig::EfficiencyBased { algo_type, .. } => algo_type.budget_config(),
        }
    }

    /*
    fn composition(&self) -> &CompositionConfig {
        match self {
            AllocationConfig::Dpf {
                block_selector_seed: _,
                composition,
            } => composition,
            AllocationConfig::Greedy { composition } => composition,
            AllocationConfig::Ilp { composition } => composition,
        }
    }
     */
}

impl CompositionConfig {
//    pub(crate) fn budget(&self) -> &BudgetTotal {
//        let temp = match self {
//            CompositionConfig::BlockCompositionPa { budget, .. } => budget,
//            CompositionConfig::BlockComposition { budget, .. } => budget,
//        };
//        match temp {
//            Budget::FixBudget { budget } => budget,
//            Budget::UnlockingBudget { budget, .. } => budget,
//        }
//    }

    pub(crate) fn budget_config(&self) -> &Budget {
        match self {
            CompositionConfig::BlockCompositionPa { budget, .. } => budget,
            CompositionConfig::BlockComposition { budget, .. } => budget,
        }
    }
}

pub fn check_output_paths(config: &Cli) -> OutputPaths {
    let req_log_output_path = config.output_config.req_log_output.clone();
    {
        // did not supply empty output_path
        let mut copy = req_log_output_path.clone();
        assert!(copy.pop(), "Empty output path was supplied");
        // all parent directories exist
        assert!(
            copy.exists(),
            "A directory on the supplied output path either does not exist or is inaccessible"
        );
        // check that file ends in .csv (via
        // https://stackoverflow.com/questions/45291832/extracting-a-file-extension-from-a-given-path-in-rust-idiomatically)
        assert_eq!(
            req_log_output_path.extension().and_then(OsStr::to_str),
            Some("csv"),
            "output file needs to have \".csv\" extension (no capital letters)"
        );
    }

    let round_log_output_path = config.output_config.round_log_output.clone();
    {
        // did not supply empty output_path
        let mut copy = round_log_output_path.clone();
        assert!(copy.pop(), "Empty output path was supplied");
        // all parent directories exist
        assert!(
            copy.exists(),
            "A directory on the supplied output path either does not exist or is inaccessible"
        );
        // check that file ends in .csv (via
        // https://stackoverflow.com/questions/45291832/extracting-a-file-extension-from-a-given-path-in-rust-idiomatically)
        assert_eq!(
            round_log_output_path.extension().and_then(OsStr::to_str),
            Some("csv"),
            "output file needs to have \".csv\" extension (no capital letters)"
        );
    }

    let runtime_log_output_path = config.output_config.runtime_log_output.clone();
    {
        // did not supply empty output_path
        let mut copy = runtime_log_output_path.clone();
        assert!(copy.pop(), "Empty output path was supplied");
        // all parent directories exist
        assert!(
            copy.exists(),
            "A directory on the supplied output path either does not exist or is inaccessible"
        );
        // check that file ends in .csv (via
        // https://stackoverflow.com/questions/45291832/extracting-a-file-extension-from-a-given-path-in-rust-idiomatically)
        assert_eq!(
            runtime_log_output_path.extension().and_then(OsStr::to_str),
            Some("csv"),
            "output file needs to have \".csv\" extension (no capital letters)"
        );
    }

    let stats_output_path = config.output_config.stats_output.clone();
    {
        // did not supply empty output_path
        let mut copy = stats_output_path.clone();
        assert!(copy.pop(), "Empty output path was supplied");
        // all parent directories exist
        assert!(
            copy.exists(),
            "A directory on the supplied output path either does not exist or is inaccessible"
        );
        // check that file ends in .csv (via
        // https://stackoverflow.com/questions/45291832/extracting-a-file-extension-from-a-given-path-in-rust-idiomatically)
        assert_eq!(
            stats_output_path.extension().and_then(OsStr::to_str),
            Some("json"),
            "output file needs to have \".json\" extension (no capital letters)"
        );
    }

    let history_output_directory_path = config.output_config.history_output_directory.clone();
    if let Some(history_path) = history_output_directory_path.as_ref() {
        // check that the given path is indeed pointing at a directory
        assert!(history_path.is_dir());
    }

    OutputPaths {
        req_log_output_path,
        round_log_output_path,
        runtime_log_output_path,
        stats_output_path,
        history_output_directory_path,
    }
}

pub struct OutputPaths {
    pub req_log_output_path: PathBuf,
    pub round_log_output_path: PathBuf,
    pub runtime_log_output_path: PathBuf,
    pub stats_output_path: PathBuf,
    pub history_output_directory_path: Option<PathBuf>,
}

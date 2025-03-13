use std::collections::HashMap;
use std::fmt;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::block::external::load_and_convert_blocks;
use crate::config::BudgetTotal;
use crate::dprivacy::privacy_unit::{MyIntervalSet, PrivacyUnit};
use crate::dprivacy::Accounting;
use crate::schema::Schema;
use crate::simulation::RoundId;
use crate::{dprivacy::AccountingType, request::Request, request::Dnf, RequestId};

pub(crate) mod external;


#[derive(Debug, Clone, Serialize)]
pub struct BudgetSection {
    pub unlocked_budget: AccountingType,
    pub total_budget: AccountingType,

    pub dnf: Dnf,
}

impl BudgetSection {
    pub fn dnf(&self) -> &Dnf {
        &self.dnf
    }
}

#[derive(Debug, Clone)]
pub struct Block {
    /// a unique identifier for this block, sometimes referred to as privacy id
    pub id: BlockId,
    /// which requests where executed on this block
    pub request_history: Vec<RequestId>,
    /// the budget that is unlocked for this block - note that this ignores the cost of executed
    /// requests but may take into account the number of executed requests
    pub default_unlocked_budget: Option<AccountingType>,

    pub default_total_budget: Option<AccountingType>,

    pub budget_by_section: Vec<BudgetSection>,

    pub privacy_unit: PrivacyUnit,
    pub privacy_unit_selection: Option<MyIntervalSet>,

    /// round when this block joins the system
    pub created: RoundId,
    /// round when this block leaves the system (i.e., is retired)
    pub retired: Option<RoundId>,
}




#[derive(Deserialize, Debug, Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub struct BlockId(pub usize);

/// Loads blocks from the specified pathbuf, and converts the budgets to rdp if alphas are set.
/// Returns both the blocks as hashmap
pub fn load_blocks(
    filepath: PathBuf,
    request_history: &HashMap<RequestId, Request>,
    schema: &Schema,
    budget_total: &BudgetTotal,
) -> Result<HashMap<BlockId, Block>, std::io::Error> {
    let blocks = load_and_convert_blocks(request_history, &schema, &budget_total, filepath)?;


    assert!(
        blocks.values().all(|bl| (bl.retired.expect("at the moment require blocks with retired time") - bl.created).to_usize() == schema.block_sliding_window_size),
        "Blocks did not have the correct sliding window size"
    );


    assert!(
        blocks
            .values()
            .all(|bl| bl.default_unlocked_budget.is_none() || schema.accounting_type.check_same_type(bl.default_unlocked_budget.as_ref().unwrap())),
        "Blocks did not have same type of DP as schema"
    );
    Ok(blocks)
}


impl fmt::Display for BlockId {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[cfg(test)]
mod tests {
    use crate::config::BudgetTotal;
    use crate::dprivacy::AccountingType::EpsDp;
    use crate::util::{build_dummy_requests_with_pa, build_dummy_schema};
    use crate::AccountingType;
    use crate::AccountingType::EpsDeltaDp;
    use std::path::PathBuf;

    #[test]
    #[should_panic(expected = "Blocks did not have same type of DP as schema")]
    fn test_loading_blocks_wrong_budget() {
        let mut schema = build_dummy_schema(EpsDp { eps: 1.0 });
        let requests = build_dummy_requests_with_pa(
            &schema,
            1,
            EpsDeltaDp {
                eps: 0.4,
                delta: 0.4,
            },
            6,
        );

        let bt = BudgetTotal::new();


        schema.accounting_type = AccountingType::EpsDeltaDp { eps: 0.0, delta: 0.0 };


        let _blocks = super::load_blocks(
            PathBuf::from("resources/test/block_files/block_test_1.json"),
            &requests,
            &schema,
            &bt,
        );
    }
}

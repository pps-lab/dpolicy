use itertools::Itertools;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Error, ErrorKind};
use std::path::PathBuf;

use crate::config::BudgetTotal;
use crate::dprivacy::privacy_unit::{MyIntervalSet, PrivacyUnit};
use crate::dprivacy::{Accounting, AdpAccounting, AccountingType};
use crate::request::external::ExternalDnf;
use crate::request::{Request, RequestId};
use crate::schema;
use crate::simulation::RoundId;
use super::{BlockId, BudgetSection};



#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ExternalBudgetSection {
    pub unlocked_budget: Option<AccountingType>,
    pub total_budget: AccountingType,
    pub dnf: ExternalDnf,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub(crate) struct ExternalBlock {
    pub(crate) id: BlockId,
    /// the requests which applied to this block
    pub(crate) request_ids: Vec<RequestId>,
    /// the budget that is unlocked for this block - note that this ignores the cost of executed
    /// requests but may take into account the number of executed requests

    /// default if nothing special is defined (in budget_by_section)
    pub(crate) unlocked_budget: Option<AccountingType>, // ignores cost of request history -> i.e., needs to be subtracted
    pub(crate) total_budget: Option<AccountingType>, // ignores cost of request history

    pub(crate) budget_by_section: Option<Vec<ExternalBudgetSection>>,


    pub privacy_unit: PrivacyUnit,
    pub privacy_unit_selection: Option<MyIntervalSet>,

    /// round when this block joins the system
    pub(crate) created: RoundId,
    /// round when this block leaves the system (i.e., is retired)
    pub(crate) retired: Option<RoundId>,
}

pub(super) fn load_and_convert_blocks(
    request_history: &HashMap<RequestId, Request>,
    schema: &schema::Schema,
    budget_total: &BudgetTotal,
    filepath: PathBuf,
) -> Result<HashMap<BlockId, super::Block>, Error> {
    let file = File::open(filepath)?;
    let reader = BufReader::new(file);

    let mut blocks: Vec<ExternalBlock> = serde_json::from_reader(reader)?;

    let target_accounting_type = &schema.accounting_type;

    blocks.iter_mut().for_each(|block| {

        // convert budgets if necessary
        if let BudgetTotal {convert_block_budgets: true, .. } = budget_total {
            let rdp_alphas = budget_total.alphas().expect("Alphas must be set to convert block budgets");

            if let Some(total_budget) = &mut block.total_budget {
                block.total_budget = Some(total_budget.adp_to_rdp_budget(&rdp_alphas));
            }

            if let Some(unlocked_budget) = &mut block.unlocked_budget {
                block.unlocked_budget = Some(unlocked_budget.adp_to_rdp_budget(&rdp_alphas));
            }

            if let Some(sections) = &mut block.budget_by_section {
                for section in sections.iter_mut() {
                    section.total_budget = section.total_budget.adp_to_rdp_budget(&rdp_alphas);
                    if let Some(unlocked_budget) = &mut section.unlocked_budget {
                        section.unlocked_budget = Some(unlocked_budget.adp_to_rdp_budget(&rdp_alphas));
                    }
                }
            }
        }

        // check default budgets on block
        if let Some(total_budget) = &block.total_budget {
            assert!(total_budget.check_same_type(target_accounting_type),
                    "Blocks total_budget did not have same type of DP as schema");

            if let Some(unlocked_budget) = &mut block.unlocked_budget {
                assert!(unlocked_budget.check_same_type(target_accounting_type),
                        "Blocks unlocked_budget did not have same type of DP as schema");
            }else{
                block.unlocked_budget = Some(AccountingType::zero_clone(total_budget));
            }
        }else{
            assert!(block.unlocked_budget.is_none(), "Total budget must be set if unlocked budget is set");
        }

        // check budget_by_sections
        if let Some(sections) = &mut block.budget_by_section {
            for section in sections.iter_mut() {

                assert!(section.total_budget.check_same_type(target_accounting_type),
                    "Blocks section total_budget did not have same type of DP as schema");

                if let Some(unlocked_budget) = &mut section.unlocked_budget {
                    assert!(unlocked_budget.check_same_type(target_accounting_type),
                            "Blocks section unlocked_budget did not have same type of DP as schema");
                }else{
                    section.unlocked_budget = Some(AccountingType::zero_clone(&section.total_budget));
                }

            }
        }

        assert!(block.total_budget.is_some() || (block.budget_by_section.as_ref().expect("if total budget is none, then this cannot be none").len() > 0), "Block must have a total budget or budget by section");
    });


    // check that there are no duplicate block ids
    let sorted_ids = blocks
        .iter()
        .map(|block| block.id)
        .sorted()
        .collect::<Vec<_>>();

    let deduped = {
        let mut temp = sorted_ids.clone();
        temp.dedup();
        temp
    };

    if deduped.len() != sorted_ids.len() {
        return Err(std::io::Error::new(ErrorKind::Other, "duplicate block ids"));
    }

    // check that all requests in each blocks history are actually in request_history
    for block in blocks.iter() {
        for request_id in block.request_ids.iter() {
            assert!(
                request_history.contains_key(request_id),
                "A request in a blocks request history was not present in overall request history"
            );
        }
    }

    Ok(blocks
        .into_iter()
        .map(|external_block| {
            (
                external_block.id,
                external_block.to_internal(schema)
            )
        })
        .collect())
}



impl ExternalBlock {

    pub fn from_internal(internal_block: &super::Block, schema: &schema::Schema) -> Self {

        let mut sections_ext: Vec<ExternalBudgetSection> = Vec::new();

        for section in internal_block.budget_by_section.iter() {
            let section_ext = ExternalBudgetSection {
                unlocked_budget: Some(section.unlocked_budget.clone()),
                total_budget: section.total_budget.clone(),
                dnf: ExternalDnf::from_internal(&section.dnf, schema),
            };
            sections_ext.push(section_ext);
        }

        let budget_by_section = if sections_ext.is_empty() {
            None
        } else {
            Some(sections_ext)
        };


        ExternalBlock {
            id: internal_block.id,
            request_ids: internal_block.request_history.clone(),
            unlocked_budget: internal_block.default_unlocked_budget.clone(),
            total_budget: internal_block.default_total_budget.clone(),
            budget_by_section,
            privacy_unit: internal_block.privacy_unit.clone(),
            privacy_unit_selection: internal_block.privacy_unit_selection.clone(),
            created: internal_block.created,
            retired: internal_block.retired,
        }
    }

    fn to_internal(self, schema: &schema::Schema) -> super::Block{

        let mut sections = Vec::new();
        if let Some(sections_ext) = &self.budget_by_section {
            for section_ext in sections_ext.iter() {


                let unlocked_budget = section_ext.unlocked_budget.clone().expect("unlocked budget must exist to convert to internal");
                let total_budget = section_ext.total_budget.clone();

                assert!(
                    unlocked_budget.check_same_type(&schema.accounting_type),
                    "blocks have incompatible budgets in budget_by_section"
                );

                assert!(
                    total_budget.check_same_type(&schema.accounting_type),
                    "blocks have incompatible budgets in budget_by_section"
                );

                let section = BudgetSection{
                    unlocked_budget: unlocked_budget,
                    total_budget: total_budget,
                    dnf: section_ext.dnf.to_internal(schema).0,
                };
                sections.push(section);

            }
        }


        self.privacy_unit.check_selection(&self.privacy_unit_selection);

        super::Block {
            id: self.id,
            request_history: self.request_ids,
            default_unlocked_budget: self.unlocked_budget,
            default_total_budget: self.total_budget,
            budget_by_section: sections,
            privacy_unit: self.privacy_unit.clone(),
            privacy_unit_selection: self.privacy_unit_selection.clone(),
            created: self.created,
            retired: self.retired,
        }
    }
}



#[cfg(test)]

mod tests {
    use crate::block::BlockId;
    use crate::config::BudgetTotal;
    use crate::request::RequestId;
    use crate::util::{build_dummy_requests_with_pa, build_dummy_schema};
    use crate::AccountingType::EpsDp;
    use crate::RoundId;
    use std::path::PathBuf;

    #[test]
    fn test_loading_blocks_1() {
        let budget = EpsDp { eps: 1.0 };
        let schema = build_dummy_schema(budget);

        let budget_total = BudgetTotal{
            privacy_units: Some(schema.privacy_units.iter().map(|x| x.clone()).collect()),
            alphas: None,
            convert_block_budgets: false,
        };

        let requests = build_dummy_requests_with_pa(&schema, 1, EpsDp { eps: 0.4 }, 6);
        let blocks = super::load_and_convert_blocks(
            &requests,
            &schema,
            &budget_total,
            PathBuf::from("resources/test/block_files/block_test_1.json"),
        );
        assert!(blocks.is_ok(), "{:?}", blocks.err());
        let block = &blocks.unwrap()[&BlockId(1)];
        assert_eq!(block.created, RoundId(0));
        assert_eq!(block.default_unlocked_budget, Some(EpsDp { eps: 0.0 }));
        assert_eq!(block.default_total_budget, Some(EpsDp { eps: 1.0 }));
        assert_eq!(block.id, BlockId(1));
        assert_eq!(
            block.request_history,
            vec![RequestId(1), RequestId(2), RequestId(3)]
        )
    }

    #[test]
    fn test_loading_blocks_2() {
        let budget = EpsDp { eps: 1.0 };
        let schema = build_dummy_schema(budget.clone());

        let budget_total = BudgetTotal{
            privacy_units: Some(schema.privacy_units.iter().map(|x| x.clone()).collect()),
            alphas: None,
            convert_block_budgets: false,
        };

        let requests = build_dummy_requests_with_pa(&schema, 1, EpsDp { eps: 0.4 }, 6);
        let blocks = super::load_and_convert_blocks(
            &requests,
            &schema,
            &budget_total,
            PathBuf::from("resources/test/block_files/block_test_2.json"),
        );
        assert!(blocks.is_ok(), "{:?}", blocks.err());
        assert_eq!(blocks.unwrap().len(), 3);
    }

    #[test]
    #[should_panic]
    fn test_loading_blocks_invalid_request_id() {
        let budget = EpsDp { eps: 1.0 };
        let schema = build_dummy_schema(budget.clone());

        let budget_total = BudgetTotal{
            privacy_units: Some(schema.privacy_units.iter().map(|x| x.clone()).collect()),
            alphas: None,
            convert_block_budgets: false,
        };

        let requests = build_dummy_requests_with_pa(&schema, 1, EpsDp { eps: 0.4 }, 6);
        let blocks = super::load_and_convert_blocks(
            &requests,
            &schema,
            &budget_total,
            PathBuf::from("resources/test/block_files/block_test_invalid_request_ids.json"),
        );
        assert!(blocks.is_ok());
    }

    #[test]
    #[should_panic]
    fn test_loading_blocks_incompatible_budgets() {
        let budget = EpsDp { eps: 1.0 };
        let schema = build_dummy_schema(budget.clone());

        let budget_total = BudgetTotal{
            privacy_units: Some(schema.privacy_units.iter().map(|x| x.clone()).collect()),
            alphas: None,
            convert_block_budgets: false,
        };
        let requests = build_dummy_requests_with_pa(&schema, 1, EpsDp { eps: 0.4 }, 6);
        let blocks = super::load_and_convert_blocks(
            &requests,
            &schema,
            &budget_total,
            PathBuf::from("resources/test/block_files/block_test_incompatible_budgets.json"),
        );
        assert!(blocks.is_ok());
    }

    #[test]
    #[should_panic]
    fn test_loading_blocks_duplicate_ids() {
        let budget = EpsDp { eps: 1.0 };
        let schema = build_dummy_schema(budget.clone());
        let requests = build_dummy_requests_with_pa(&schema, 1, EpsDp { eps: 0.4 }, 6);

        let budget_total = BudgetTotal{
            privacy_units: Some(schema.privacy_units.iter().map(|x| x.clone()).collect()),
            alphas: None,
            convert_block_budgets: false,
        };
        let blocks = super::load_and_convert_blocks(
            &requests,
            &schema,
            &budget_total,
            PathBuf::from("resources/test/block_files/block_test_duplicate_ids.json"),
        );
        assert!(blocks.is_ok());
    }
}

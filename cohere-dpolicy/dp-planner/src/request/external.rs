use crate::dprivacy::privacy_unit::{MyIntervalSet, PrivacyUnit};
use crate::dprivacy::Accounting;
use crate::simulation::RoundId;
use crate::AccountingType;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashMap, HashSet};
use std::fs::File;
use std::io::{BufReader, Error};
use std::path::PathBuf;

use super::super::schema;
use super::super::schema::DataValueLookup;
use super::{AttributeId, ConjunctionBuilder, Dnf, RequestBuilder, RequestId};

/// ExternalRequest is the serialized format of [super::Request].
/// However, there also a few semantic
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ExternalRequest {
    pub request_id: RequestId,
    pub request_cost: HashMap<PrivacyUnit, AccountingType>,
    pub privacy_unit_selection: HashMap<PrivacyUnit, MyIntervalSet>,
    pub profit: u64,
    pub dnf: ExternalDnf,

    pub dnf_pa: Option<ExternalDnf>,
    pub attributes: Option<Vec<String>>,
    pub categories: Option<Vec<String>>,
    pub relaxations: Option<Vec<String>>,

    /// identifies the round the request joins the system
    pub created: RoundId,
}

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct ExternalDnf {
    pub(super) conjunctions: Vec<Conjunction>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Conjunction {
    pub(crate) predicates: HashMap<String, Predicate>, //Key: Name of attribute, value: Predicate on that attribute
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum Predicate {
    // TODO [later]: Expand with Gt, Le etc
    Eq(schema::DataValue),
    Neq(schema::DataValue),
    // TODO [later] not sure whether min max should be doable with non number
    Between {
        min: schema::DataValue,
        max: schema::DataValue,
    }, //including both ends
    In(HashSet<schema::DataValue>), // Note: Can contain different variants of DataValue at the same time, e.g., String("xy") and Integer(3)
}

trait ExternalPredicate<'a> {
    fn to_internal(
        &self,
        attr_name: &str,
        schema: &schema::Schema,
    ) -> Result<(super::AttributeId, super::Predicate), schema::SchemaError>;
}


impl ExternalDnf {

    pub fn from_internal(dnf: &Dnf, schema: &schema::Schema) -> Self {

        let confjunctions = dnf
            .conjunctions
            .iter()
            .map(|conj| {
                let predicates: HashMap<String, Predicate> = conj
                    .predicates
                    .iter()
                    .enumerate()
                    .map(|(attr_id, pred)| {
                        (
                            schema
                                .attribute_name(attr_id)
                                .expect("Couldn't get attribute name")
                                .to_string(),
                            pred.to_external(AttributeId(attr_id), schema)
                                .expect("Couldn't convert predicate to external"),
                        )
                    })
                    .collect();

                Conjunction { predicates }
            })
            .collect();

        ExternalDnf{
            conjunctions: confjunctions,
        }
    }



    pub fn to_internal(&self, schema: &schema::Schema) -> (super::Dnf, Vec<String>) {

        let mut dnf = super::Dnf {
            conjunctions: Vec::new(),
        };

        let mut missing_attributes: Vec<String> = Vec::new();
        // convert external dnf -> internal dnf
        for external_conjunction in self.conjunctions.iter() {
            let mut conj_builder = ConjunctionBuilder::new(schema);

            // convert predicates from external -> internal
            for (name, external_pred) in external_conjunction.predicates.iter() {
                match external_pred.to_internal(name, schema) {
                    Ok(sol) => {
                        let (attr_id, pred) = sol;
                        conj_builder = conj_builder.and(attr_id, pred);
                    }
                    Err(_) => missing_attributes.push(name.to_string()),
                }
            }

            dnf.conjunctions.push(conj_builder.build());
        }

        (dnf, missing_attributes)
    }
}

impl<'a> ExternalPredicate<'a> for Predicate {
    fn to_internal(
        &self,
        attr_name: &str,
        schema: &schema::Schema,
    ) -> Result<(super::AttributeId, super::Predicate), schema::SchemaError> {
        let attr_id = schema.attribute_id(attr_name)?;
        match self {
            Predicate::Eq(val) => {
                let val = schema.attribute_idx(attr_id, val)?;
                Ok((super::AttributeId(attr_id), super::Predicate::Eq(val)))
            }
            Predicate::Neq(val) => {
                let val = schema.attribute_idx(attr_id, val)?;
                Ok((super::AttributeId(attr_id), super::Predicate::Neq(val)))
            }
            Predicate::Between { min, max } => {
                let min = schema.attribute_idx(attr_id, min)?;
                let max = schema.attribute_idx(attr_id, max)?;
                Ok((
                    super::AttributeId(attr_id),
                    super::Predicate::Between { min, max },
                ))
            }
            Predicate::In(vals) => {
                let values: Result<HashSet<usize>, schema::SchemaError> = vals
                    .iter()
                    .map(|x| schema.attribute_idx(attr_id, x))
                    .collect();
                match values {
                    Ok(values) => Ok((super::AttributeId(attr_id), super::Predicate::In(values))),
                    Err(e) => Err(e),
                }
            }
        }
    }
}

impl ExternalRequest {
    /// Returns the converted request, and a list of attributes which could not be converted
    fn to_internal(&self, schema: &schema::Schema) -> (super::Request, Vec<String>) {

        let request_cost: HashMap<PrivacyUnit, AccountingType> = schema.privacy_units.iter().map(|unit| (unit.clone(), self.request_cost.get(unit).expect("missing privacy unit for request").clone())).collect();

        let mut builder = RequestBuilder::new_full(
            self.request_id,
            request_cost,
            self.privacy_unit_selection.clone(),
            self.profit,
            None,
            self.created,
            schema,
        );

        // convert external dnf -> internal dnf

        let (internal_dnf, missing_attributes) = self.dnf.to_internal(schema);

        for conj in internal_dnf.conjunctions.into_iter(){
            builder = builder.or_conjunction(conj);
        }

        (builder.build(), missing_attributes)
    }
}

pub fn parse_requests(filepath: PathBuf) -> Result<Vec<ExternalRequest>, Error> {
    let file = File::open(filepath)?;
    let reader = BufReader::new(file);

    // Read the JSON contents of the file as an instance of `User`.
    let requests: Vec<ExternalRequest> = serde_json::from_reader(reader).expect("Parsing Failed");

    //println!("{:?}", request);

    Ok(requests)
}

pub fn convert_requests(
    requests: Vec<ExternalRequest>,
    schema: &schema::Schema,
) -> Result<HashMap<RequestId, super::Request>, schema::SchemaError> {


    // then we convert it to internal representation
    let mut internal_requests: HashMap<RequestId, super::Request> = HashMap::new();
    let mut all_missing_attributes: BTreeSet<String> = BTreeSet::new();


    for r in requests.into_iter() {
        let (converted, missing_attributes) = r.to_internal(schema);

        let inserted = internal_requests.insert(converted.request_id, converted);
        assert!(inserted.is_none());
        all_missing_attributes.extend(missing_attributes.into_iter());
    }

    // finally, check if cost in schema and the requests is of the same type
    assert!(
        internal_requests
            .values()
            .all(|req| req.request_cost.values().all(|cost| cost.check_same_type(&schema.accounting_type))),
        "Request and schema have different types of DP. Schema: {}, first request: {:?}",
        schema.accounting_type,
        internal_requests.values().next().unwrap().request_cost
    );

    if !all_missing_attributes.is_empty() {
        println!(
            "Requests had attributes which were not part of schema: {:?}",
            all_missing_attributes
        )
    }

    Ok(internal_requests)
}

#[cfg(test)]
mod tests {
    use crate::config::BudgetTotal;
    use crate::dprivacy::privacy_unit::PrivacyUnit;
    use crate::request::{external::parse_requests, load_requests, resource_path};


    static DEMO_REQUESTS: &str = "request_files/demo_requests.json";
    static CENSUS_REQUESTS: &str = "request_files/census_requests.json";
    static DEMO_SCHEMA: &str = "schema_files/demo_schema.json";
    static CENSUS_SCHEMA: &str = "schema_files/census_schema.json";


    #[test]
    fn test_parse_demo_requests() {
        let demo_requests = parse_requests(resource_path(DEMO_REQUESTS));
        assert!(demo_requests.is_ok());
    }

    #[test]
    fn test_parse_census_requests() {
        let census_requests = parse_requests(resource_path(CENSUS_REQUESTS));
        assert!(census_requests.is_ok());
    }

    #[test]
    fn test_convert_demo_requests() {

        let budget = BudgetTotal{
            privacy_units: Some(vec![PrivacyUnit::User]),
            alphas: Some(vec![0.; 5]),
            convert_block_budgets: false,
        };


        let demo_schema =
            crate::schema::load_schema(resource_path(DEMO_SCHEMA), &budget, Some(1)).unwrap();
        let converted_request = load_requests(
            resource_path(DEMO_REQUESTS),
            &demo_schema,
        );
        assert!(
            converted_request.is_ok(),
            "{}",
            converted_request.err().unwrap().to_string()
        );
    }

    #[test]
    fn test_convert_census_requests() {

        let budget = BudgetTotal{
            privacy_units: Some(vec![PrivacyUnit::User]),
            alphas: Some(vec![0.; 13]),
            convert_block_budgets: false,
        };
        let census_schema = crate::schema::load_schema(
            resource_path(CENSUS_SCHEMA),
            &budget,
            Some(1),
        )
        .unwrap();
        let converted_request = load_requests(
            resource_path(CENSUS_REQUESTS),
            &census_schema,
        );
        assert!(
            converted_request.is_ok(),
            "{}",
            converted_request.err().unwrap().to_string()
        );
    }

    #[test]
    #[should_panic(expected = "Request and schema have different types of DP")]
    fn test_convert_census_requests_wrong_budget() {
        let budget = BudgetTotal{
            privacy_units: Some(vec![PrivacyUnit::User]),
            alphas: Some(vec![0.; 10]),
            convert_block_budgets: false,
        };
        let census_schema = crate::schema::load_schema(
            resource_path(CENSUS_SCHEMA),
            &budget,
            Some(1),
        )
        .unwrap();
        let converted_request = load_requests(
            resource_path(CENSUS_REQUESTS),
            &census_schema,
        );
        assert!(
            converted_request.is_ok(),
            "{}",
            converted_request.err().unwrap().to_string()
        );
    }
}

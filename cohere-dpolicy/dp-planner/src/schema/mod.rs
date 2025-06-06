use crate::config::BudgetTotal;
use crate::dprivacy::privacy_unit::PrivacyUnit;
use crate::dprivacy::rdp_alphas_accounting::{PubRdpAccounting, RdpAlphas};
use crate::request::{ConjunctionBuilder, RequestBuilder};
use crate::AccountingType::EpsDp;
use crate::RequestId;
use itertools::{Itertools, MultiProduct};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::fs::File;
use std::io::{BufReader, Error};
use std::ops::Range;
use std::path::PathBuf;

use super::dprivacy::AccountingType;



// TODO [later]: should be called something like blocking schema, accounting schema or something like that to avoid confusion with "full data schema"
#[derive(Clone)]
pub struct Schema {
    pub accounting_type: AccountingType,

    pub privacy_units: HashSet<PrivacyUnit>,

    pub block_sliding_window_size: usize,

    pub attributes: Vec<Attribute>,

    pub name_to_index: HashMap<String, usize>,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct ExternalSchema {
    accounting_type: Option<AccountingType>,
    privacy_units: Option<HashSet<PrivacyUnit>>,
    block_sliding_window_size: Option<usize>,

    attributes: Vec<Attribute>,
    attributes_pa: Option<Vec<Attribute>>,

    #[serde(skip)]
    name_to_index: HashMap<String, usize>,
}

pub trait DataValueLookup {
    /// Get attribute id corresponding to a certain attribute name. Opposite of
    /// [Self::attribute_name]
    fn attribute_id(&self, name: &str) -> Result<usize, SchemaError>;

    /// Get the attributename corresponding to a certain attribute id. Opposite of
    /// [Self::attribute_id]
    fn attribute_name(&self, attribute_id: usize) -> Result<&str, SchemaError>;

    /// Given an attribute id (of the schema) and attribute value, returns the index of this
    /// value (i.e., transforms the value to an usize in [0, N[ where N is the number of values
    /// this attribute can have). Can reverse with [Self::attribute_value]
    fn attribute_idx(
        &self,
        attribute_id: usize,
        attribute_value: &DataValue,
    ) -> Result<usize, SchemaError>;

    /// Basically the reverse function of [Self::attribute_idx]: Given an attribute id (relating to
    /// the schema) and a normalised value (i.e., an index into the set of values
    /// allowable for this attribute) for this attribute, returns the "real" value.
    fn attribute_value(
        &self,
        attribute_id: usize,
        attribute_idx: usize,
    ) -> Result<DataValue, SchemaError>;

    fn virtual_block_id_iterator(&self) -> MultiProduct<Range<usize>>;
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Attribute {
    pub name: String,
    pub value_domain: ValueDomain,

    #[serde(skip)]
    pub value_domain_map: Option<HashMap<DataValue, usize>>, // TODO [later] make private again
}

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(untagged)]
pub enum ValueDomain {
    Range { min: isize, max: isize },
    Collection(Vec<DataValue>),
}

#[derive(Serialize, Deserialize, Clone, PartialEq, Eq, Hash, Debug)]
#[serde(untagged)]
pub enum DataValue {
    Integer(isize),
    String(String),
    Bool(bool),
}

#[derive(Debug, Clone)]
pub struct SchemaError(pub String);

pub fn load_schema(filepath: PathBuf, budget_total: &BudgetTotal, block_sliding_window_size: Option<usize>) -> Result<Schema, Error> {
    let file = File::open(filepath)?;
    let reader = BufReader::new(file);

    // Read the JSON contents of the file as an instance of `User`.
    let external_schema: ExternalSchema = serde_json::from_reader(reader).expect("Parsing Failed");

    let schema = Schema::to_internal(budget_total, block_sliding_window_size, external_schema);

    Ok(schema)
    //println!("{:?}", request);
}

pub fn load_schema_pa(filepath: PathBuf, budget_total: &BudgetTotal, block_sliding_window_size: Option<usize>) -> Result<Schema, Error> {
    let file = File::open(filepath)?;
    let reader = BufReader::new(file);

    // Read the JSON contents of the file as an instance of `User`.
    let mut external_schema: ExternalSchema = serde_json::from_reader(reader).expect("Parsing Failed");

    // we replace the attributes with the attributes_pa
    external_schema.attributes = external_schema.attributes_pa.clone().expect("No attributes_pa given in schema");



    let schema = Schema::to_internal(budget_total, block_sliding_window_size, external_schema);

    Ok(schema)
}


impl Schema {

    pub fn to_internal(budget_total: &BudgetTotal, block_sliding_window_size: Option<usize>, external_schema: ExternalSchema) -> Schema {

        let accounting_type = match &budget_total.alphas {
            Some(rdp) => AccountingType::Rdp { eps_values: RdpAlphas::from_vec(vec![0.0; rdp.len()]).expect("not supported rdp length") },
            None => external_schema.accounting_type.expect("No accounting type given (neither in cli nor in schema)"),
        };

        let privacy_units =  match &budget_total.privacy_units {
            Some(privacy_units) => privacy_units.iter().map(|x| x.clone()).collect(),
            None => external_schema.privacy_units.expect("No privacy units given (neither in cli nor in schema)"),
        };

        let block_sliding_window_size = match block_sliding_window_size {
            Some(block_sliding_window_size) => block_sliding_window_size,
            None => external_schema.block_sliding_window_size.expect("At the moment, only the block sliding window approach is supported (but was not given neither in cli nor in schema"),
        };

        let mut schema = Schema {
            accounting_type,
            privacy_units,
            block_sliding_window_size,
            attributes: external_schema.attributes,
            name_to_index: external_schema.name_to_index,
        };

        schema.init();

        println!("Schema: accounting_type={:?}  privacy_units={:?} block_sliding_window={:?} virtual_blocks_schema={:?}", &schema.accounting_type, &schema.privacy_units, &schema.block_sliding_window_size, &schema.attributes);

        schema
    }



    /// Converts an internal schema to external, to serialize or later use as input
    pub fn to_external(&self) -> ExternalSchema {
        ExternalSchema {
            accounting_type: Some(self.accounting_type.clone()),
            privacy_units: Some(self.privacy_units.clone()),
            attributes: self.attributes.clone(),
            attributes_pa: None,
            block_sliding_window_size: Some(self.block_sliding_window_size),
            name_to_index: Default::default(),
        }
    }
}

impl DataValueLookup for Schema {
    fn attribute_id(&self, name: &str) -> Result<usize, SchemaError> {
        self.name_to_index
            .get(name)
            .copied()
            .ok_or_else(|| SchemaError(format!("Invalid Attribute Name: {}  (not found)", name)))
    }

    fn attribute_name(&self, attribute_id: usize) -> Result<&str, SchemaError> {
        self.attributes
            .get(attribute_id)
            .map(|x| x.name.as_str())
            .ok_or_else(|| {
                SchemaError(format!(
                    "Invalid Attribute Id: {}  (not found)",
                    attribute_id
                ))
            })
    }

    fn attribute_idx(
        &self,
        attribute_id: usize,
        attribute_value: &DataValue,
    ) -> Result<usize, SchemaError> {
        let attribute = self.attributes.get(attribute_id).ok_or_else(|| {
            SchemaError(format!(
                "Invalid Attribute Id: {}  (not found)",
                attribute_id
            ))
        })?;
        attribute.value_to_index(attribute_value)
    }

    fn attribute_value(
        &self,
        attribute_id: usize,
        attribute_idx: usize,
    ) -> Result<DataValue, SchemaError> {
        let attribute = self.attributes.get(attribute_id).ok_or_else(|| {
            SchemaError(format!(
                "Invalid Attribute Id: {}  (not found)",
                attribute_id
            ))
        })?;
        attribute.index_to_value(attribute_idx)
    }

    fn virtual_block_id_iterator(&self) -> MultiProduct<Range<usize>> {
        let mut ranges: Vec<usize> = Vec::with_capacity(self.attributes.len());
        for attr in self.attributes.iter() {
            ranges.push(attr.len())
        }
        ranges
            .into_iter()
            .map(|len| 0..len)
            .multi_cartesian_product()
    }
}

impl Schema {
    pub fn init(&mut self) {
        //check that there are no duplicate attribute names
        let mut attribute_names: HashSet<String> = HashSet::new();
        for attr in self.attributes.iter() {
            let inserted = attribute_names.insert(attr.name.clone());
            assert!(
                inserted,
                "Schema invalid, attribute name \"{}\" given twice",
                attr.name
            );
        }

        // init each attribute (build internal data structure for faster lookup)
        for (index, attr) in self.attributes.iter_mut().enumerate() {
            attr.init();
            self.name_to_index.insert(attr.name.clone(), index);
        }
    }

    /// This function calculates the number of virtual blocks in the schema passed as &self.
    pub fn num_virtual_blocks(&self) -> usize {
        // construct a request that has all virtual blocks from the schema (achieved by
        // adding just one one conjunction which is empty -> all virtual blocks match)
        let full_request = RequestBuilder::new(
            RequestId(0),
            HashMap::from([(PrivacyUnit::User, EpsDp { eps: 0.0 })]),
            HashMap::new(),
            0,
            None,
            self,
        )
        .or_conjunction(ConjunctionBuilder::new(self).build())
        .build();

        // now, #virtual blocks of this request must be = #virtual blocks of schema -> can reuse
        // method from request

        full_request.dnf().num_virtual_blocks(self)
    }
}

impl Attribute {
    fn init(&mut self) {
        // builds a lookup such that given a DataValue, we can figure out the internal index
        self.value_domain_map = match &self.value_domain {
            // ValueDomain::Range{min, max} => Some((*min..*max).enumerate().map(|(idx, x)| (DataValue::Integer(x), idx)).collect()),
            ValueDomain::Range { .. } => None, // not really needed for ranges
            ValueDomain::Collection(vec) => Some(
                vec.iter()
                    .enumerate()
                    .map(|(idx, x)| (x.clone(), idx))
                    .collect(),
            ),
        }
    }

    pub fn len(&self) -> usize {
        match &self.value_domain {
            ValueDomain::Range { min, max } => ((max - min) + 1) as usize,
            ValueDomain::Collection(vec) => vec.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn value_to_index(&self, value: &DataValue) -> Result<usize, SchemaError> {
        match (&self.value_domain, &self.value_domain_map, value) {
            (ValueDomain::Range { min, max }, None, DataValue::Integer(num))
                if num >= min && num <= max =>
            {
                Ok((num - min) as usize)
            }
            (ValueDomain::Collection(_vec), Some(lookup), value) => {
                lookup.get(value).copied().ok_or_else(|| {
                    SchemaError(format!("Value not in Domain: {:?}  (Collection)", value))
                })
            }
            (ValueDomain::Collection(_), None, _) => Err(SchemaError(
                "attribute.value_domain_map must be build for collection -> call attribute.init()"
                    .to_string(),
            )),
            _ => Err(SchemaError(format!(
                "Value {:?} not in Domain {:?}, likely due to invalid predicate.",
                value, self.value_domain
            ))),
        }
    }

    fn index_to_value(&self, index: usize) -> Result<DataValue, SchemaError> {
        match &self.value_domain {
            ValueDomain::Range { min, max } if min + index as isize <= *max => {
                Ok(DataValue::Integer(min + index as isize))
            }
            ValueDomain::Collection(vec) if index < vec.len() => Ok(vec[index].clone()),
            _ => Err(SchemaError(format!("Index not in Domain: {:?} ", index))),
        }
    }
}

impl fmt::Display for SchemaError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "schema error {}", self.0)
    }
}

/*

*/

#[cfg(test)]
mod tests {
    use crate::config::BudgetTotal;
    use crate::dprivacy::privacy_unit::PrivacyUnit;
    use crate::request::resource_path;
    use crate::schema::{load_schema, ExternalSchema};
    use std::fs::File;
    use std::io::{BufReader, Error};
    use std::path::PathBuf;
    use std::str::FromStr;

    static DEMO_SCHEMA: &str = "schema_files/demo_schema.json";
    static CENSUS_SCHEMA: &str = "schema_files/census_schema.json";
    static DEMO_SCHEMA_DUPL_ATTR: &str = "schema_files/demo_schema_duplicate_attribute.json";

    fn parse_schema(filepath: &str) -> Result<ExternalSchema, Error> {
        let file = File::open(filepath)?;
        let reader = BufReader::new(file);

        // Read the JSON contents of the file as an instance of `User`.
        let requests: ExternalSchema = serde_json::from_reader(reader).expect("Parsing Failed");

        //println!("{:?}", request);

        Ok(requests)
    }

    #[test]
    fn test_parse_census_schema() {
        let census_schema = parse_schema(resource_path(CENSUS_SCHEMA).to_str().unwrap());
        assert!(census_schema.is_ok());
    }

    #[test]
    fn test_init_census_schema() {

        let budget = BudgetTotal{
            privacy_units: Some(vec![PrivacyUnit::User]),
            alphas: Some(vec![0.0; 13]),
            convert_block_budgets: false,
        };


        let census_schema = load_schema(
            PathBuf::from_str(resource_path(CENSUS_SCHEMA).to_str().unwrap())
                .expect("Parsing path failed"),
            &budget,
            Some(1),
        );
        census_schema.unwrap().init();
    }

    #[test]
    fn test_parse_demo_schema() {
        let demo_schema = parse_schema(resource_path(DEMO_SCHEMA).to_str().unwrap());
        assert!(demo_schema.is_ok());
    }

    #[test]
    fn test_init_demo_schema() {

        let budget = BudgetTotal{
            privacy_units: Some(vec![PrivacyUnit::User]),
            alphas: Some(vec![0.0; 13]),
            convert_block_budgets: false,
        };

        let demo_schema = load_schema(
            PathBuf::from_str(resource_path(DEMO_SCHEMA).to_str().unwrap())
                .expect("Parsing path failed"),
            &budget,
            Some(1),
        );
        demo_schema.unwrap().init()
    }

    #[test]
    fn test_census_schema_num_virtual_blocks() {

        let budget = BudgetTotal{
            privacy_units: Some(vec![PrivacyUnit::User]),
            alphas: Some(vec![0.0; 13]),
            convert_block_budgets: false,
        };


        let mut census_schema = load_schema(
            PathBuf::from_str(resource_path(CENSUS_SCHEMA).to_str().unwrap())
                .expect("Parsing path failed"),
            &budget,
            Some(1),
        )
        .unwrap();
        census_schema.init();
        assert_eq!(census_schema.num_virtual_blocks(), 153600)
    }

    #[test]
    fn test_parse_demo_schema_duplicate_attribute() {
        let demo_schema = parse_schema(resource_path(DEMO_SCHEMA_DUPL_ATTR).to_str().unwrap());
        assert!(demo_schema.is_ok());
    }

    #[test]
    #[should_panic]
    fn test_init_demo_schema_duplicate_attribute() {

        let budget = BudgetTotal{
            privacy_units: Some(vec![PrivacyUnit::User]),
            alphas: None,
            convert_block_budgets: false,
        };

        let demo_schema = load_schema(
            PathBuf::from_str(resource_path(DEMO_SCHEMA_DUPL_ATTR).to_str().unwrap())
                .expect("Parsing path failed"),
            &budget,
            Some(1),
        );
        demo_schema.unwrap().init();
    }
}

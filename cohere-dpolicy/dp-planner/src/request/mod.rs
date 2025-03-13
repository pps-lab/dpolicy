//! Contains structs and methods to define and handle requests.
//!
//! Functions to load and convert external requests, as well as the
//! external request definition is part of [external], everything relating to request adapters in
//! [adapter], and [internal] contains methods for handling requests inside this program

pub mod external;
pub mod internal;


use std::{
    collections::{BTreeMap, HashMap, HashSet},
    fmt,
    path::{Path, PathBuf},
};
use bincode::{Decode, Encode};
use itertools::MultiProduct;
use serde::{Deserialize, Serialize};

use crate::{dprivacy::privacy_unit::{MyIntervalSet, PrivacyUnit}, schema::{DataValue, DataValueLookup}};
use crate::simulation::RoundId;
use crate::{
    schema::{Schema, SchemaError},
    AccountingType,
};


use self::internal::PredicateWithSchemaIntoIterator;

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, Deserialize, PartialOrd, Ord, Serialize, Encode, Decode)]
pub struct RequestId(pub usize);

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct AttributeId(pub usize);

#[allow(dead_code)]
pub fn resource_path(filename: &str) -> PathBuf {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"));
    path.join("resources").join("test").join(filename)
}

pub fn load_requests(
    request_path: PathBuf,
    schema: &Schema,
) -> Result<HashMap<RequestId, Request>, SchemaError> {
    let external_requests =
        external::parse_requests(request_path).expect("Failed to open or parse requests");
    external::convert_requests(external_requests, schema)
}


pub fn load_requests_pa(
    request_path: PathBuf,
    schema: &Schema,
) -> (HashMap<RequestId, Request>, BTreeMap<String, HashSet<RequestId>>, BTreeMap<String, HashSet<RequestId>>, BTreeMap<String, HashSet<RequestId>>) {
    let mut external_requests =
        external::parse_requests(request_path).expect("Failed to open or parse requests");

    let mut attribute_lookup = BTreeMap::new();

    let mut category_lookup = BTreeMap::new();

    let mut relaxation_lookup = BTreeMap::new();


    // use only the dnf_pa as the dnf
    for r in external_requests.iter_mut() {

        r.dnf = r.dnf_pa.clone().expect("No dnf_pa given in request");

        let attributes = r.attributes.clone().expect("No attributes given in request");
        for a in attributes.into_iter() {
            attribute_lookup.entry(a.clone())
                        .or_insert_with(HashSet::new)
                        .insert(r.request_id);
        }

        let categories = r.categories.clone().expect("No attributes given in request");
        for c in categories.into_iter() {
            category_lookup.entry(c.clone())
                        .or_insert_with(HashSet::new)
                        .insert(r.request_id);
        }

        let relaxations = r.relaxations.clone().expect("No attributes given in request");
        for relax in relaxations.into_iter() {
            relaxation_lookup.entry(relax.clone())
                        .or_insert_with(HashSet::new)
                        .insert(r.request_id);
        }


    }

    let requests = external::convert_requests(external_requests, schema).unwrap();

    (requests, attribute_lookup, category_lookup, relaxation_lookup)

}

impl Dnf {

    pub fn repeating_iter<'a, 'b>(&'a self, schema: &'b Schema) -> DNFRepeatingIterator<'a, 'b> {
        let iters: Vec<_> = self
            .conjunctions
            .iter()
            .map(|conj| conj.prod_iter(schema))
            .collect();
        assert!(!iters.is_empty());
        DNFRepeatingIterator {
            conj_index: 0,
            iterators: iters,
        }
    }

    pub fn num_virtual_blocks(&self, schema: &Schema) -> usize {
        let virtual_blocks: HashSet<Vec<usize>> = self.repeating_iter(schema).collect();
        virtual_blocks.len()
    }
}

#[derive(Clone, Debug)]
pub struct Request {
    /// A unique identifier for this request. Often used as the key for some hash- or tree-based
    /// map to access a request object.
    pub request_id: RequestId,
    /// How much cost this request incurs when it is allocated some blocks / segments of a block.
    request_cost: HashMap<PrivacyUnit, AccountingType>,

    pub privacy_unit_selection: HashMap<PrivacyUnit, MyIntervalSet>,

    /// How important this request is compared to other requests. Often times, the goal of
    /// [crate::allocation] is to maximize the total profit
    pub profit: u64,
    dnf: Dnf,
    pub num_blocks: Option<usize>,
    /// identifies the round the request joins the system. Default: 0
    pub created: RoundId,
}

/// Records whether certain fields in a request were changed by an adapter, and if these changes
/// were named, also records that name. Needed for evaluation purposes.
#[derive(Default, Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub struct AdapterInfo {
    privacy_cost: ModificationStatus,
    n_users: ModificationStatus,
    profit: ModificationStatus,
}

/// Records whether a certain field in the record was changed by the privacy adapter, as well as
/// the name of the applied adapter option, if this was specified
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum ModificationStatus {
    /// The field was not changed by the adapter
    Unchanged,
    /// The field was changed by the adapter, but the rule that changed it does not
    /// have a name.
    Unnamed,
    /// The field was changed by the adapter, and the rule is named as specified.
    Named(String),
}

impl Default for ModificationStatus {
    fn default() -> Self {
        Self::Unchanged
    }
}

/// Used to start building a request. Usually initialized via [RequestBuilder::new],
/// and then further modified via [RequestBuilder::or_conjunction], and the finalized request
/// is extracted via [RequestBuilder::build]
pub struct RequestBuilder<'a> {
    /// The schema to which the request adheres. The attributes as well as their
    /// respective ranges should be a superset of the ones in the request.
    schema: &'a Schema,
    /// The current state of the request which is being built.
    request: Request,
}

impl fmt::Display for RequestId {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl<'a> RequestBuilder<'a> {
    /// This initializes a [RequestBuilder] with the given arguments. Calling this method and
    /// then adding conjunctions via [RequestBuilder::or_conjunction] is preferable to manually
    /// creating requests, as this is less error prone and easier to read.
    pub fn new(
        request_id: RequestId,
        request_cost: HashMap<PrivacyUnit, AccountingType>,
        privacy_unit_selection: HashMap<PrivacyUnit, MyIntervalSet>,
        profit: u64,
        num_blocks: Option<usize>,
        schema: &'a Schema,
    ) -> Self {
        let created = RoundId(0);
        RequestBuilder::new_full(request_id, request_cost, privacy_unit_selection, profit, num_blocks, created, schema)
    }

    pub fn new_full(
        request_id: RequestId,
        request_cost: HashMap<PrivacyUnit, AccountingType>,
        privacy_unit_selection: HashMap<PrivacyUnit, MyIntervalSet>,
        profit: u64,
        num_blocks: Option<usize>,
        created: RoundId,
        schema: &'a Schema,
    ) -> Self {

        request_cost.iter().for_each(|(unit, _)|  unit.check_selection(&privacy_unit_selection.get(unit).map(|x| x.clone())));

        let request = Request {
            request_id,
            request_cost,
            privacy_unit_selection,
            profit,
            dnf: Dnf {
                conjunctions: Vec::new(),
            },
            num_blocks,
            created,
        };

        RequestBuilder { request, schema }
    }

    /// Add another conjunction to the dnf in the request in the [RequestBuilder]
    pub fn or_conjunction(mut self, c: Conjunction) -> Self {
        // TODO [nku] DELETE AND REPLACE WITH INTERNAL TO EXTERNAL MAYBE?
        self.request.dnf.conjunctions.push(c);
        self
    }

    /// Construct the request from the input given to [Self::new] and [Self::or_conjunction].
    pub fn build(mut self) -> Request {
        if self.request.dnf.conjunctions.is_empty() {
            // request with empty dnf -> want all blokcs => need to insert all blocks conjunction
            let all_conjunction = ConjunctionBuilder::new(self.schema).build();
            self.request.dnf.conjunctions.push(all_conjunction);
        }

        self.request
    }
}

/// The disjunctive normal form of the predicates attached to a request.
#[derive(Clone, Debug, Serialize)]
pub struct Dnf {
    /// Each entry is a conjunction, together making up the disjunctive normal form of the request
    /// predicates.
    pub conjunctions: Vec<Conjunction>,
}

/// An iterator which visits each virtual field that is part of a requests demand. Note that this
/// iterator may visit a virtual field multiple times.
#[derive(Clone)]
pub struct DNFRepeatingIterator<'a, 'b> {
    /// Which conjunction we are currently looking at
    conj_index: usize,
    /// An iterator for each conjunction
    iterators: Vec<MultiProduct<PredicateWithSchemaIntoIterator<'a, 'b>>>,
}

/// The basic building block for the disjunctive normal form which defines request predicates.
///
/// Is part of [Dnf]
#[derive(Clone, Debug, Serialize)]
pub struct Conjunction {
    // TODO [later]: a small inefficiency is that each conjunction needs to contain a predicate for all attributes.
    // We could change this and store the "full" predicate once per schema.
    // However, this results to problems with lifetimes in Conjunction::prod_iter(..)
    /// A vector of predicates, which for each attribute defines the values that match to the given
    /// request.
    ///
    /// That each conjunction needs to contain a predicate for all attributes
    /// us a small inefficiency.
    /// We could change this and store the "full" predicate once per schema.
    /// However, this results to problems with lifetimes in Conjunction::prod_iter(..)
    predicates: Vec<Predicate>, // must always include a predicate for all values in schema
}

/// Preferred way to initialise conjunctions manually.
///
/// First, a [ConjunctionBuilder] should be initialized with [ConjunctionBuilder::new], then
/// new predicates should be appended with [ConjunctionBuilder::and], and then finally, the
/// conjunction should be built with [ConjunctionBuilder::build]
pub struct ConjunctionBuilder<'a> {
    schema: &'a Schema,
    predicates: HashMap<AttributeId, Predicate>,
}

/// A predicate defines a set of acceptable values for a certain attribute.
///
/// Note that the attribute to which a predicate belongs is not stored here, and is part of the
/// datastructure which the predicate is part of.
#[derive(Clone, Debug, Serialize)]
pub enum Predicate {
    // TODO [later]: Expand with Gt, Le etc
    /// The predicate is only true if the attribute takes the given value.
    Eq(usize),
    /// The predicate is only true if the attribute does NOT take the given value
    Neq(usize),
    /// The predicate is only true if the attribute is between min and max (including both ends)
    Between { min: usize, max: usize }, //including both ends
    /// The predicate is only true if the attribute takes a value which is part of the given set.
    In(HashSet<usize>),
}

impl Predicate {
    /// This method "translates" a predicate from a [Request] to a predicate of an
    /// [external::ExternalRequest].
    ///
    /// This is useful if one wants to serialize a [Request], which requires transforming the it to
    /// an [external::ExternalRequest] first.
    fn to_external(
        &self,
        attribute_id: AttributeId,
        schema: &Schema,
    ) -> Result<external::Predicate, SchemaError> {
        match self {
            Predicate::Eq(x) => {
                let val = schema.attribute_value(attribute_id.0, *x)?;
                Ok(external::Predicate::Eq(val))
            }
            Predicate::Neq(x) => {
                let val = schema.attribute_value(attribute_id.0, *x)?;
                Ok(external::Predicate::Neq(val))
            }
            Predicate::Between { min, max } => {
                let min = schema.attribute_value(attribute_id.0, *min)?;
                let max = schema.attribute_value(attribute_id.0, *max)?;
                Ok(external::Predicate::Between { min, max })
            }
            Predicate::In(x) => Ok(external::Predicate::In(
                x.iter()
                    .map(|y| schema.attribute_value(attribute_id.0, *y))
                    .collect::<Result<HashSet<DataValue>, SchemaError>>()?,
            )),
        }
    }
}

/// This datastructure provides context for a predicate. Provides an
/// into_iter method to iterate over all values part of the schema for which
/// the predicate evaluates to true.
pub struct PredicateWithSchema<'a, 'b> {
    /// The id of the attribute to which the predicate belongs
    attribute_id: AttributeId,
    /// A reference to the predicate in question
    predicate: &'a Predicate,
    /// A reference to the schema which defines the current partitioning attributes.
    schema: &'b Schema,
}

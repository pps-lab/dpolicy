use gcollections::ops::Bounded;
use interval::{interval_set::ToIntervalSet, IntervalSet};
use serde::de::{self, Visitor};
use serde::{Deserialize, Serialize, Deserializer, Serializer};
use core::panic;
use std::fmt;
use std::ops::{Deref, DerefMut};
use gcollections::ops::set::Overlap;
use interval::ops::Range;

use crate::block::Block;
use crate::request::Request;

#[derive(Deserialize, Eq, Hash, Clone, Debug, PartialEq, PartialOrd, Serialize, clap::ArgEnum, Ord)]
pub enum PrivacyUnit {
    User,
    UserDay,
    UserWeek,
    UserMonth,
    UserYear,
    User100Year,
}

impl fmt::Display for PrivacyUnit {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let s = match self {
            PrivacyUnit::UserDay => "UserDay",
            PrivacyUnit::UserWeek => "UserWeek",
            PrivacyUnit::UserMonth => "UserMonth",
            PrivacyUnit::UserYear => "UserYear",
            PrivacyUnit::User100Year => "User100Year",
            PrivacyUnit::User => "User",
        };
        write!(f, "{}", s)
    }
}

impl PrivacyUnit {
    pub fn check_selection(&self, selection: &Option<MyIntervalSet>) {
        match (self, selection) {
            (PrivacyUnit::User, None) => true,
            (PrivacyUnit::UserDay, Some(_)) => true,
            (PrivacyUnit::UserWeek, Some(_)) => true,
            (PrivacyUnit::UserMonth, Some(_)) => true,
            (PrivacyUnit::UserYear, Some(_)) => true,
            (PrivacyUnit::User100Year, Some(_)) => true,
            _ => panic!("Invalid privacy unit and selection combination: {:?}", (self, selection)),
        };
    }



}


pub fn is_selected_block(request: &Request, block: &Block) -> bool{
    let is_selected = match &block.privacy_unit{
        PrivacyUnit::User => true,
        _ => {
            let request_selection = request.privacy_unit_selection.get(&block.privacy_unit).expect("Privacy unit not found in request");

            let block_selection = block.privacy_unit_selection.as_ref().expect("Block privacy unit selection not set");

            block_selection.0.overlap(&request_selection.0)
        }
    };
    is_selected
}

#[derive(Debug, Clone, PartialEq)]
pub struct MyIntervalSet(IntervalSet<u64>);


impl MyIntervalSet {
    pub fn new(low: u64, high: u64) -> Self {
        MyIntervalSet(IntervalSet::new(low, high))
    }
}


impl Deref for MyIntervalSet {
    type Target = IntervalSet<u64>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for MyIntervalSet {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl Serialize for MyIntervalSet {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let res: Vec<(u64, u64)> = self
            .iter()
            .map(|interval| (interval.lower(), interval.upper()))
            .collect();
        res.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for MyIntervalSet {
    fn deserialize<D>(deserializer: D) -> Result<MyIntervalSet, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct MyIntervalSetVisitor;

        impl<'de> Visitor<'de> for MyIntervalSetVisitor {
            type Value = MyIntervalSet;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("a sequence of (u64, u64) tuples")
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: de::SeqAccess<'de>,
            {
                let mut vec = Vec::new();

                while let Some((first, second)) = seq.next_element::<(u64, u64)>()? {
                    vec.push((first, second));
                }

                Ok(MyIntervalSet(vec.to_interval_set()))
            }
        }

        deserializer.deserialize_seq(MyIntervalSetVisitor)
    }
}

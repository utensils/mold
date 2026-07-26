use std::collections::BTreeMap;
use std::ops::{Add, Neg};

use crate::{CandidatePlacement, DeviceId, WorkId};

#[derive(Clone, Debug)]
pub(crate) struct MatchCandidate {
    pub placement: CandidatePlacement,
    pub setup_ms: u64,
}

#[derive(Clone, Debug)]
pub(crate) struct MatchWork {
    pub work_id: WorkId,
    pub candidates: Vec<MatchCandidate>,
}

#[derive(Clone, Debug)]
pub(crate) struct MatchingResult {
    pub assignments: BTreeMap<WorkId, MatchCandidate>,
    pub total_host_ram_bytes: u64,
}

#[derive(Clone, Copy, Debug, Default, Eq, Ord, PartialEq, PartialOrd)]
struct Cost {
    host_ram: i128,
    setup: i128,
    stable_tie: i128,
}

impl Add for Cost {
    type Output = Self;

    fn add(self, right: Self) -> Self::Output {
        Self {
            host_ram: self.host_ram + right.host_ram,
            setup: self.setup + right.setup,
            stable_tie: self.stable_tie + right.stable_tie,
        }
    }
}

impl Neg for Cost {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self {
            host_ram: -self.host_ram,
            setup: -self.setup,
            stable_tie: -self.stable_tie,
        }
    }
}

#[derive(Clone, Debug)]
struct Edge {
    to: usize,
    reverse: usize,
    capacity: u8,
    cost: Cost,
    candidate_index: Option<usize>,
}

fn add_edge(
    graph: &mut [Vec<Edge>],
    from: usize,
    to: usize,
    cost: Cost,
    candidate_index: Option<usize>,
) {
    let forward_reverse = graph[to].len();
    let reverse_reverse = graph[from].len();
    graph[from].push(Edge {
        to,
        reverse: forward_reverse,
        capacity: 1,
        cost,
        candidate_index,
    });
    graph[to].push(Edge {
        to: from,
        reverse: reverse_reverse,
        capacity: 0,
        cost: -cost,
        candidate_index: None,
    });
}

/// Incremental deterministic min-cost bipartite matcher.
///
/// Work is visited in hard priority order. A tentative insertion clones at
/// most a `2 * device_count + 2` node residual graph, adds one work vertex,
/// and runs exactly one successive augmenting path. Accepting the clone keeps
/// the prior minimum-cost flow, including reverse edges that globally rematch
/// flexible work around a newly admitted specialist.
#[derive(Clone, Debug)]
pub(crate) struct IncrementalMatcher {
    device_nodes: BTreeMap<DeviceId, usize>,
    work_capacity: usize,
    first_work: usize,
    first_device: usize,
    source: usize,
    sink: usize,
    graph: Vec<Vec<Edge>>,
    works: Vec<MatchWork>,
}

impl IncrementalMatcher {
    pub fn new(device_ids: &[DeviceId]) -> Self {
        let source = 0;
        let first_work = 1;
        let work_capacity = device_ids.len();
        let first_device = first_work + work_capacity;
        let sink = first_device + device_ids.len();
        let node_count = sink + 1;
        let mut graph = vec![Vec::new(); node_count];
        let device_nodes = device_ids
            .iter()
            .enumerate()
            .map(|(index, id)| (id.clone(), first_device + index))
            .collect::<BTreeMap<_, _>>();
        for device_index in 0..device_ids.len() {
            add_edge(
                &mut graph,
                first_device + device_index,
                sink,
                Cost::default(),
                None,
            );
        }
        Self {
            device_nodes,
            work_capacity,
            first_work,
            first_device,
            source,
            sink,
            graph,
            works: Vec::new(),
        }
    }

    pub fn try_with(&self, work: MatchWork) -> Option<(Self, MatchingResult)> {
        if self.works.len() >= self.work_capacity {
            return None;
        }
        let mut trial = self.clone();
        let work_node = trial.first_work + trial.works.len();
        add_edge(
            &mut trial.graph,
            trial.source,
            work_node,
            Cost::default(),
            None,
        );
        let mut candidates = work.candidates.iter().enumerate().collect::<Vec<_>>();
        candidates.sort_by(|(left_index, left), (right_index, right)| {
            left.placement
                .device_id
                .cmp(&right.placement.device_id)
                .then_with(|| {
                    left.placement
                        .incremental_host_ram_bytes
                        .cmp(&right.placement.incremental_host_ram_bytes)
                })
                .then_with(|| left.setup_ms.cmp(&right.setup_ms))
                .then_with(|| left_index.cmp(right_index))
        });
        for (candidate_index, candidate) in candidates {
            let device_node = *trial.device_nodes.get(&candidate.placement.device_id)?;
            let stable_tie = (device_node - trial.first_device) as i128;
            add_edge(
                &mut trial.graph,
                work_node,
                device_node,
                Cost {
                    host_ram: i128::from(candidate.placement.incremental_host_ram_bytes),
                    setup: i128::from(candidate.setup_ms),
                    stable_tie,
                },
                Some(candidate_index),
            );
        }
        trial.works.push(work);

        let node_count = trial.graph.len();
        let mut distance = vec![None::<Cost>; node_count];
        let mut predecessor = vec![None::<(usize, usize)>; node_count];
        distance[trial.source] = Some(Cost::default());

        for _ in 0..node_count.saturating_sub(1) {
            let mut changed = false;
            for from in 0..node_count {
                let Some(from_distance) = distance[from] else {
                    continue;
                };
                for (edge_index, edge) in trial.graph[from].iter().enumerate() {
                    if edge.capacity == 0 {
                        continue;
                    }
                    let candidate_distance = from_distance + edge.cost;
                    if distance[edge.to].is_none_or(|current| candidate_distance < current) {
                        distance[edge.to] = Some(candidate_distance);
                        predecessor[edge.to] = Some((from, edge_index));
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }

        predecessor[trial.sink]?;
        let mut node = trial.sink;
        while node != trial.source {
            let (from, edge_index) = predecessor[node]?;
            let reverse = trial.graph[from][edge_index].reverse;
            trial.graph[from][edge_index].capacity = 0;
            trial.graph[node][reverse].capacity = 1;
            node = from;
        }

        let matching = trial.result()?;
        Some((trial, matching))
    }

    fn result(&self) -> Option<MatchingResult> {
        let mut assignments = BTreeMap::new();
        let mut total_host_ram_bytes = 0_u64;
        for (work_index, work) in self.works.iter().enumerate() {
            let work_node = self.first_work + work_index;
            let matched_candidate = self.graph[work_node]
                .iter()
                .filter_map(|edge| {
                    if edge.capacity == 0 && edge.to >= self.first_device && edge.to < self.sink {
                        edge.candidate_index
                    } else {
                        None
                    }
                })
                .next()?;
            let candidate = work.candidates[matched_candidate].clone();
            total_host_ram_bytes =
                total_host_ram_bytes.checked_add(candidate.placement.incremental_host_ram_bytes)?;
            assignments.insert(work.work_id.clone(), candidate);
        }
        Some(MatchingResult {
            assignments,
            total_host_ram_bytes,
        })
    }
}

# Credit Assignment via Interaction Combinators
# The causal graph for RL credit assignment implemented as an interaction net
#
# Key insight: Credit assignment graphs ARE interaction nets
# - State nodes = Wires (carry information)
# - Action nodes = Agents (active nodes that transform)
# - Influence weights = Port annotations
# - Fan-in (shared cause) = γ node (constructor)
# - Fan-out (multiple effects) = δ node (duplicator)
# - Credit propagation = Backward reduction (dual graph)
# - Graph compression = Interaction rules (annihilation/commutation)

# ============================================================
# Core Tensor Type (shared with rest of Paragon)
# ============================================================

type Tensor:
  Node { ~left: Tensor, ~right: Tensor }
  Leaf { val: f24 }
  Nil

# ============================================================
# Credit Assignment Node Types
# ============================================================

# State node: holds environment state embedding at a timestep
type StateNode:
  State {
    time: u24,              # Timestep in trajectory
    embedding: Tensor,      # State representation
    credit: f24             # Accumulated credit (filled during backward pass)
  }

# Action node: represents a policy decision point
type ActionNode:
  Action {
    time: u24,              # Timestep when action taken
    logits: Tensor,         # Policy output logits
    chosen: u24,            # Index of chosen action
    log_prob: f24,          # Log probability of chosen action
    credit: f24             # Accumulated credit (filled during backward pass)
  }

# Reward node: terminal credit source
type RewardNode:
  Reward {
    time: u24,              # Timestep when reward received
    value: f24              # Reward value (source of credit)
  }

# ============================================================
# Interaction Combinator Nodes (γ and δ)
# ============================================================

# Gamma node (γ): Constructor - combines causes into effect
# In credit assignment: action + state → next_state
# When credit flows backward: credit splits to causes weighted by influence
type GammaNode:
  Gamma {
    id: u24,                # Unique node identifier
    weight: f24,            # Influence weight (learned)
    principal: u24,         # ID of principal port (effect)
    aux1: u24,              # ID of auxiliary port 1 (cause 1)
    aux2: u24,              # ID of auxiliary port 2 (cause 2)
    credit: f24             # Credit flowing through this node
  }

# Delta node (δ): Duplicator - one cause has multiple effects
# In credit assignment: one state/action influences multiple future states
# When credit flows backward: credits sum from all effects
type DeltaNode:
  Delta {
    id: u24,                # Unique node identifier
    split_ratio: f24,       # How credit splits (0.0 to 1.0)
    principal: u24,         # ID of principal port (cause)
    aux1: u24,              # ID of auxiliary port 1 (effect 1)
    aux2: u24,              # ID of auxiliary port 2 (effect 2)
    credit: f24             # Credit flowing through this node
  }

# ============================================================
# Influence Edge (weighted causal connection)
# ============================================================

type InfluenceEdge:
  Influence {
    id: u24,                # Edge identifier
    weight: f24,            # Causal influence weight (0.0 to 1.0)
    from_node: u24,         # Source node ID
    to_node: u24,           # Target node ID
    from_type: NodeType,    # Type of source node
    to_type: NodeType       # Type of target node
  }

type NodeType:
  TypeState
  TypeAction
  TypeReward
  TypeGamma
  TypeDelta

# ============================================================
# Credit Assignment Graph (Interaction Net)
# ============================================================

# List of state nodes
type StateList:
  SCons { head: StateNode, ~tail: StateList }
  SNil

# List of action nodes
type ActionList:
  ACons { head: ActionNode, ~tail: ActionList }
  ANil

# List of reward nodes
type RewardList:
  RCons { head: RewardNode, ~tail: RewardList }
  RNil

# List of gamma nodes
type GammaList:
  GCons { head: GammaNode, ~tail: GammaList }
  GNil

# List of delta nodes
type DeltaList:
  DCons { head: DeltaNode, ~tail: DeltaList }
  DNil

# List of influence edges
type EdgeList:
  ECons { head: InfluenceEdge, ~tail: EdgeList }
  ENil

# The complete credit assignment graph
type CreditGraph:
  Graph {
    states: StateList,      # All state nodes
    actions: ActionList,    # All action nodes
    rewards: RewardList,    # All reward nodes (credit sources)
    gammas: GammaList,      # Constructor nodes
    deltas: DeltaList,      # Duplicator nodes
    edges: EdgeList,        # Weighted influence edges
    next_id: u24            # Counter for generating unique IDs
  }

# ============================================================
# Graph Construction
# ============================================================

# Create empty credit graph
def credit_graph_new() -> CreditGraph:
  return CreditGraph/Graph {
    states: StateList/SNil,
    actions: ActionList/ANil,
    rewards: RewardList/RNil,
    gammas: GammaList/GNil,
    deltas: DeltaList/DNil,
    edges: EdgeList/ENil,
    next_id: 0
  }

# Add a state node to the graph
def graph_add_state(graph: CreditGraph, time: u24, embedding: Tensor) -> (CreditGraph, u24):
  match graph:
    case CreditGraph/Graph:
      node_id = graph.next_id
      new_state = StateNode/State {
        time: time,
        embedding: embedding,
        credit: 0.0
      }
      new_graph = CreditGraph/Graph {
        states: StateList/SCons { head: new_state, tail: graph.states },
        actions: graph.actions,
        rewards: graph.rewards,
        gammas: graph.gammas,
        deltas: graph.deltas,
        edges: graph.edges,
        next_id: graph.next_id + 1
      }
      return (new_graph, node_id)

# Add an action node to the graph
def graph_add_action(graph: CreditGraph, time: u24, logits: Tensor, chosen: u24, log_prob: f24) -> (CreditGraph, u24):
  match graph:
    case CreditGraph/Graph:
      node_id = graph.next_id
      new_action = ActionNode/Action {
        time: time,
        logits: logits,
        chosen: chosen,
        log_prob: log_prob,
        credit: 0.0
      }
      new_graph = CreditGraph/Graph {
        states: graph.states,
        actions: ActionList/ACons { head: new_action, tail: graph.actions },
        rewards: graph.rewards,
        gammas: graph.gammas,
        deltas: graph.deltas,
        edges: graph.edges,
        next_id: graph.next_id + 1
      }
      return (new_graph, node_id)

# Add a reward node (credit source)
def graph_add_reward(graph: CreditGraph, time: u24, value: f24) -> (CreditGraph, u24):
  match graph:
    case CreditGraph/Graph:
      node_id = graph.next_id
      new_reward = RewardNode/Reward {
        time: time,
        value: value
      }
      new_graph = CreditGraph/Graph {
        states: graph.states,
        actions: graph.actions,
        rewards: RewardList/RCons { head: new_reward, tail: graph.rewards },
        gammas: graph.gammas,
        deltas: graph.deltas,
        edges: graph.edges,
        next_id: graph.next_id + 1
      }
      return (new_graph, node_id)

# Add a gamma node (constructor: combines action + state → next_state)
def graph_add_gamma(graph: CreditGraph, weight: f24, principal: u24, aux1: u24, aux2: u24) -> (CreditGraph, u24):
  match graph:
    case CreditGraph/Graph:
      node_id = graph.next_id
      new_gamma = GammaNode/Gamma {
        id: node_id,
        weight: weight,
        principal: principal,
        aux1: aux1,
        aux2: aux2,
        credit: 0.0
      }
      new_graph = CreditGraph/Graph {
        states: graph.states,
        actions: graph.actions,
        rewards: graph.rewards,
        gammas: GammaList/GCons { head: new_gamma, tail: graph.gammas },
        deltas: graph.deltas,
        edges: graph.edges,
        next_id: graph.next_id + 1
      }
      return (new_graph, node_id)

# Add a delta node (duplicator: one cause → multiple effects)
def graph_add_delta(graph: CreditGraph, split_ratio: f24, principal: u24, aux1: u24, aux2: u24) -> (CreditGraph, u24):
  match graph:
    case CreditGraph/Graph:
      node_id = graph.next_id
      new_delta = DeltaNode/Delta {
        id: node_id,
        split_ratio: split_ratio,
        principal: principal,
        aux1: aux1,
        aux2: aux2,
        credit: 0.0
      }
      new_graph = CreditGraph/Graph {
        states: graph.states,
        actions: graph.actions,
        rewards: graph.rewards,
        gammas: graph.gammas,
        deltas: DeltaList/DCons { head: new_delta, tail: graph.deltas },
        edges: graph.edges,
        next_id: graph.next_id + 1
      }
      return (new_graph, node_id)

# Add an influence edge between nodes
def graph_add_edge(graph: CreditGraph, weight: f24, from_node: u24, to_node: u24, from_type: NodeType, to_type: NodeType) -> CreditGraph:
  match graph:
    case CreditGraph/Graph:
      edge_id = graph.next_id
      new_edge = InfluenceEdge/Influence {
        id: edge_id,
        weight: weight,
        from_node: from_node,
        to_node: to_node,
        from_type: from_type,
        to_type: to_type
      }
      return CreditGraph/Graph {
        states: graph.states,
        actions: graph.actions,
        rewards: graph.rewards,
        gammas: graph.gammas,
        deltas: graph.deltas,
        edges: EdgeList/ECons { head: new_edge, tail: graph.edges },
        next_id: graph.next_id + 1
      }

# ============================================================
# Interaction Rules (Credit Flow)
# ============================================================

# Rule 1: γ-γ annihilation (two constructors meet)
# Credit merges when shared causes combine
# Returns the combined weight
def gamma_gamma_rule(g1: GammaNode, g2: GammaNode) -> f24:
  match g1:
    case GammaNode/Gamma:
      match g2:
        case GammaNode/Gamma:
          # Weights multiply through the path
          return g1.weight * g2.weight

# Rule 2: δ-δ annihilation (two duplicators meet)
# Credit sums when multiple effects combine
def delta_delta_rule(d1: DeltaNode, d2: DeltaNode) -> f24:
  match d1:
    case DeltaNode/Delta:
      match d2:
        case DeltaNode/Delta:
          # Credits from both paths add together
          return d1.credit + d2.credit

# Rule 3: γ-δ commutation (constructor meets duplicator)
# This is where credit redistribution happens
# Returns (new_weight1, new_weight2) for the two new paths
def gamma_delta_rule(g: GammaNode, d: DeltaNode) -> (f24, f24):
  match g:
    case GammaNode/Gamma:
      match d:
        case DeltaNode/Delta:
          # Credit splits proportionally through new structure
          w1 = g.weight * d.split_ratio
          w2 = g.weight * (1.0 - d.split_ratio)
          return (w1, w2)

# ============================================================
# Backward Credit Propagation (Graph Reduction)
# ============================================================

# Inject credit at a reward node
def inject_credit_reward(reward: RewardNode) -> f24:
  match reward:
    case RewardNode/Reward:
      return reward.value

# Propagate credit through a gamma node (backward)
# Credit splits to aux ports weighted by influence
def propagate_gamma_backward(gamma: GammaNode, incoming_credit: f24) -> (f24, f24):
  match gamma:
    case GammaNode/Gamma:
      # Credit to aux1 (typically state)
      credit1 = incoming_credit * gamma.weight
      # Credit to aux2 (typically action)
      credit2 = incoming_credit * gamma.weight
      return (credit1, credit2)

# Propagate credit through a delta node (backward)
# Credits from multiple effects sum at the cause
def propagate_delta_backward(delta: DeltaNode, credit1: f24, credit2: f24) -> f24:
  match delta:
    case DeltaNode/Delta:
      # Weighted sum of credits from both effects
      return credit1 * delta.split_ratio + credit2 * (1.0 - delta.split_ratio)

# ============================================================
# Credit Accumulation in Parallel Trees
# ============================================================

# Tree structure for parallel credit accumulation
type CreditTree:
  CNode { ~left: CreditTree, ~right: CreditTree }
  CLeaf { time: u24, credit: f24 }
  CNil

# Build credit tree from action list (for parallel reduction)
def build_credit_tree(actions: ActionList) -> CreditTree:
  match actions:
    case ActionList/ACons:
      match actions.head:
        case ActionNode/Action:
          return CreditTree/CNode {
            left: CreditTree/CLeaf { time: actions.head.time, credit: actions.head.credit },
            right: build_credit_tree(actions.tail)
          }
    case ActionList/ANil:
      return CreditTree/CNil

# Parallel credit accumulation
def accumulate_credits(tree: CreditTree, incoming: f24, weight: f24) -> CreditTree:
  match tree:
    case CreditTree/CNode:
      # Parallel: update both branches simultaneously
      return CreditTree/CNode {
        left: accumulate_credits(tree.left, incoming, weight),
        right: accumulate_credits(tree.right, incoming, weight)
      }
    case CreditTree/CLeaf:
      new_credit = tree.credit + incoming * weight
      return CreditTree/CLeaf { time: tree.time, credit: new_credit }
    case CreditTree/CNil:
      return CreditTree/CNil

# Sum all credits in tree (parallel reduction)
def sum_credits(tree: CreditTree) -> f24:
  fold tree:
    case CreditTree/CNode:
      return tree.left + tree.right
    case CreditTree/CLeaf:
      return tree.credit
    case CreditTree/CNil:
      return 0.0

# ============================================================
# Credit Normalization (Conservation Law)
# ============================================================

# Normalize credits to sum to total reward (credit conservation)
def normalize_credits(tree: CreditTree, total_reward: f24) -> CreditTree:
  total_credit = sum_credits(tree)
  if total_credit > 0.0001:
    scale = total_reward / total_credit
    return scale_credits(tree, scale)
  else:
    return tree

def scale_credits(tree: CreditTree, scale: f24) -> CreditTree:
  match tree:
    case CreditTree/CNode:
      return CreditTree/CNode {
        left: scale_credits(tree.left, scale),
        right: scale_credits(tree.right, scale)
      }
    case CreditTree/CLeaf:
      return CreditTree/CLeaf { time: tree.time, credit: tree.credit * scale }
    case CreditTree/CNil:
      return CreditTree/CNil

# ============================================================
# Credit Map (Time → Credit) for Policy Gradient
# ============================================================

# Map from timestep to credit value
type CreditMap:
  CMapCons { time: u24, credit: f24, ~tail: CreditMap }
  CMapNil

# Convert credit tree to map
def credit_tree_to_map(tree: CreditTree) -> CreditMap:
  match tree:
    case CreditTree/CNode:
      left_map = credit_tree_to_map(tree.left)
      right_map = credit_tree_to_map(tree.right)
      return credit_map_merge(left_map, right_map)
    case CreditTree/CLeaf:
      return CreditMap/CMapCons { time: tree.time, credit: tree.credit, tail: CreditMap/CMapNil }
    case CreditTree/CNil:
      return CreditMap/CMapNil

def credit_map_merge(a: CreditMap, b: CreditMap) -> CreditMap:
  match a:
    case CreditMap/CMapCons:
      return CreditMap/CMapCons {
        time: a.time,
        credit: a.credit,
        tail: credit_map_merge(a.tail, b)
      }
    case CreditMap/CMapNil:
      return b

# Lookup credit for a given timestep
def credit_map_get(map: CreditMap, time: u24) -> f24:
  match map:
    case CreditMap/CMapCons:
      if map.time == time:
        return map.credit
      else:
        return credit_map_get(map.tail, time)
    case CreditMap/CMapNil:
      return 0.0

# ============================================================
# Graph Statistics (for debugging/monitoring)
# ============================================================

def count_states(states: StateList) -> u24:
  match states:
    case StateList/SCons:
      return 1 + count_states(states.tail)
    case StateList/SNil:
      return 0

def count_actions(actions: ActionList) -> u24:
  match actions:
    case ActionList/ACons:
      return 1 + count_actions(actions.tail)
    case ActionList/ANil:
      return 0

def count_rewards(rewards: RewardList) -> u24:
  match rewards:
    case RewardList/RCons:
      return 1 + count_rewards(rewards.tail)
    case RewardList/RNil:
      return 0

def count_edges(edges: EdgeList) -> u24:
  match edges:
    case EdgeList/ECons:
      return 1 + count_edges(edges.tail)
    case EdgeList/ENil:
      return 0

def graph_size(graph: CreditGraph) -> (u24, u24, u24, u24):
  match graph:
    case CreditGraph/Graph:
      n_states = count_states(graph.states)
      n_actions = count_actions(graph.actions)
      n_rewards = count_rewards(graph.rewards)
      n_edges = count_edges(graph.edges)
      return (n_states, n_actions, n_rewards, n_edges)

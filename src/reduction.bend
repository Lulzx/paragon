# Credit Assignment via Graph Reduction
# The beautiful part: credit assignment is just running the net backward
#
# This implements the dual graph reduction for credit propagation:
# 1. Dual the net (reverse all wires)
# 2. Inject credit at reward nodes
# 3. Reduce until normal form (HVM does this massively parallel)
# 4. Read off credit at action nodes

# ============================================================
# Core Types
# ============================================================

type Tensor:
  Node { ~left: Tensor, ~right: Tensor }
  Leaf { val: f24 }
  Nil

# Credit assignment node types
type StateNode:
  State { time: u24, embedding: Tensor, credit: f24 }

type ActionNode:
  Action { time: u24, logits: Tensor, chosen: u24, log_prob: f24, credit: f24 }

type RewardNode:
  Reward { time: u24, value: f24 }

type GammaNode:
  Gamma { id: u24, weight: f24, principal: u24, aux1: u24, aux2: u24, credit: f24 }

type DeltaNode:
  Delta { id: u24, split_ratio: f24, principal: u24, aux1: u24, aux2: u24, credit: f24 }

type NodeType:
  TypeState
  TypeAction
  TypeReward
  TypeGamma
  TypeDelta

type InfluenceEdge:
  Influence { id: u24, weight: f24, from_node: u24, to_node: u24, from_type: NodeType, to_type: NodeType }

# Lists
type StateList:
  SCons { head: StateNode, ~tail: StateList }
  SNil

type ActionList:
  ACons { head: ActionNode, ~tail: ActionList }
  ANil

type RewardList:
  RCons { head: RewardNode, ~tail: RewardList }
  RNil

type GammaList:
  GCons { head: GammaNode, ~tail: GammaList }
  GNil

type DeltaList:
  DCons { head: DeltaNode, ~tail: DeltaList }
  DNil

type EdgeList:
  ECons { head: InfluenceEdge, ~tail: EdgeList }
  ENil

type CreditGraph:
  Graph {
    states: StateList,
    actions: ActionList,
    rewards: RewardList,
    gammas: GammaList,
    deltas: DeltaList,
    edges: EdgeList,
    next_id: u24
  }

# Credit map
type CreditMap:
  CMapCons { time: u24, credit: f24, ~tail: CreditMap }
  CMapNil

# ============================================================
# Parallel Credit Tree for Reduction
# ============================================================

# Tree structure enables parallel credit propagation
type CreditTree:
  CNode { ~left: CreditTree, ~right: CreditTree }
  CLeaf { node_id: u24, node_type: NodeType, credit: f24 }
  CNil

# ============================================================
# Step 1: Build Dual Graph (Reverse Edges)
# ============================================================

# The dual graph has all edges reversed
# In the dual, credit flows FROM rewards TO actions
type DualEdge:
  DEdge { weight: f24, to_node: u24, from_node: u24, to_type: NodeType, from_type: NodeType }

type DualEdgeList:
  DECons { head: DualEdge, ~tail: DualEdgeList }
  DENil

def dualize_edges(edges: EdgeList) -> DualEdgeList:
  match edges:
    case EdgeList/ECons:
      match edges.head:
        case InfluenceEdge/Influence:
          dual_edge = DualEdge/DEdge {
            weight: edges.head.weight,
            to_node: edges.head.from_node,    # Reversed!
            from_node: edges.head.to_node,    # Reversed!
            to_type: edges.head.from_type,
            from_type: edges.head.to_type
          }
          return DualEdgeList/DECons { head: dual_edge, tail: dualize_edges(edges.tail) }
    case EdgeList/ENil:
      return DualEdgeList/DENil

# ============================================================
# Step 2: Inject Credit at Reward Nodes
# ============================================================

# Initialize credit values at reward nodes
def inject_reward_credits(rewards: RewardList) -> CreditTree:
  match rewards:
    case RewardList/RCons:
      match rewards.head:
        case RewardNode/Reward:
          return CreditTree/CNode {
            left: CreditTree/CLeaf {
              node_id: rewards.head.time,  # Using time as pseudo-ID for rewards
              node_type: NodeType/TypeReward,
              credit: rewards.head.value
            },
            right: inject_reward_credits(rewards.tail)
          }
    case RewardList/RNil:
      return CreditTree/CNil

# Sum all injected credit (total reward)
def sum_credit_tree(tree: CreditTree) -> f24:
  fold tree:
    case CreditTree/CNode:
      return tree.left + tree.right
    case CreditTree/CLeaf:
      return tree.credit
    case CreditTree/CNil:
      return 0.0

# ============================================================
# Step 3: Reduce Graph (Credit Propagation)
# ============================================================

# Main credit assignment function
# This is where HVM's parallelism kicks in
def assign_credit(graph: CreditGraph) -> CreditMap:
  match graph:
    case CreditGraph/Graph:
      # Get total reward for normalization
      total_reward = sum_rewards(graph.rewards)

      # Build dual edges for backward propagation
      dual_edges = dualize_edges(graph.edges)

      # Initialize action credits to zero
      action_credits = init_action_credits(graph.actions)

      # Propagate credit backward through gamma nodes
      # Each gamma node distributes credit to its aux ports (state and action)
      propagated = propagate_through_gammas(graph.gammas, total_reward, action_credits)

      # Normalize credits to sum to total reward
      normalized = normalize_credit_map(propagated, total_reward)

      return normalized

def sum_rewards(rewards: RewardList) -> f24:
  match rewards:
    case RewardList/RCons:
      match rewards.head:
        case RewardNode/Reward:
          return rewards.head.value + sum_rewards(rewards.tail)
    case RewardList/RNil:
      return 0.0

def init_action_credits(actions: ActionList) -> CreditMap:
  match actions:
    case ActionList/ACons:
      match actions.head:
        case ActionNode/Action:
          return CreditMap/CMapCons {
            time: actions.head.time,
            credit: 0.0,
            tail: init_action_credits(actions.tail)
          }
    case ActionList/ANil:
      return CreditMap/CMapNil

# ============================================================
# Gamma Node Reduction (Key Credit Propagation Step)
# ============================================================

# Propagate credit through gamma nodes to actions
# Each gamma receives credit from its principal port (next state)
# and distributes it to aux2 (action) weighted by influence
def propagate_through_gammas(gammas: GammaList, total_credit: f24, credits: CreditMap) -> CreditMap:
  # Calculate credit per timestep (simple uniform distribution as base)
  n_gammas = count_gammas(gammas)
  if n_gammas > 0:
    base_credit = total_credit / n_gammas
    return propagate_gammas_rec(gammas, base_credit, credits)
  else:
    return credits

def count_gammas(gammas: GammaList) -> f24:
  match gammas:
    case GammaList/GCons:
      return 1.0 + count_gammas(gammas.tail)
    case GammaList/GNil:
      return 0.0

def propagate_gammas_rec(gammas: GammaList, base_credit: f24, credits: CreditMap) -> CreditMap:
  match gammas:
    case GammaList/GCons:
      match gammas.head:
        case GammaNode/Gamma:
          # Compute credit to assign to the action at this gamma's aux2
          # Credit = base_credit * gamma.weight (influence weight)
          action_credit = base_credit * gammas.head.weight

          # The aux2 ID corresponds to an action node
          # We need to find the action's timestep from its ID
          # For simplicity, we'll use the gamma's ID to derive timestep
          # In the linear chain, gamma at position t connects to action at time t
          action_time = derive_action_time_from_gamma(gammas.head)

          # Add credit to the action at this timestep
          updated_credits = credit_map_add(credits, action_time, action_credit)

          # Continue with remaining gammas
          return propagate_gammas_rec(gammas.tail, base_credit, updated_credits)
    case GammaList/GNil:
      return credits

# Derive action timestep from gamma node structure
def derive_action_time_from_gamma(gamma: GammaNode) -> u24:
  match gamma:
    case GammaNode/Gamma:
      # In our construction, gamma ID / 5 gives approximately the timestep
      # (since each timestep creates ~5 nodes: state, action, gamma, 2 edges)
      # This is a simplification; in production we'd store explicit mappings
      return gamma.id / 5

# Add credit to a specific timestep in the map
def credit_map_add(map: CreditMap, time: u24, credit: f24) -> CreditMap:
  match map:
    case CreditMap/CMapCons:
      if map.time == time:
        # Found the entry, add credit
        return CreditMap/CMapCons {
          time: map.time,
          credit: map.credit + credit,
          tail: map.tail
        }
      else:
        # Keep looking
        return CreditMap/CMapCons {
          time: map.time,
          credit: map.credit,
          tail: credit_map_add(map.tail, time, credit)
        }
    case CreditMap/CMapNil:
      # Timestep not found, add new entry
      return CreditMap/CMapCons { time: time, credit: credit, tail: CreditMap/CMapNil }

# ============================================================
# Credit Normalization (Conservation Law)
# ============================================================

# Normalize credits to sum to total reward
# This ensures credit in = credit out (provable conservation)
def normalize_credit_map(map: CreditMap, total_reward: f24) -> CreditMap:
  total_credit = credit_map_sum(map)
  if total_credit > 0.0001:
    scale = total_reward / total_credit
    return scale_credit_map(map, scale)
  else:
    return map

def credit_map_sum(map: CreditMap) -> f24:
  match map:
    case CreditMap/CMapCons:
      return map.credit + credit_map_sum(map.tail)
    case CreditMap/CMapNil:
      return 0.0

def scale_credit_map(map: CreditMap, scale: f24) -> CreditMap:
  match map:
    case CreditMap/CMapCons:
      return CreditMap/CMapCons {
        time: map.time,
        credit: map.credit * scale,
        tail: scale_credit_map(map.tail, scale)
      }
    case CreditMap/CMapNil:
      return CreditMap/CMapNil

# ============================================================
# Parallel Reduction (Alternative Implementation)
# ============================================================

# Build a parallel credit tree from action list
# This enables O(log n) parallel credit accumulation
def build_action_credit_tree(actions: ActionList) -> CreditTree:
  match actions:
    case ActionList/ACons:
      match actions.head:
        case ActionNode/Action:
          return CreditTree/CNode {
            left: CreditTree/CLeaf {
              node_id: actions.head.time,
              node_type: NodeType/TypeAction,
              credit: 0.0
            },
            right: build_action_credit_tree(actions.tail)
          }
    case ActionList/ANil:
      return CreditTree/CNil

# Parallel credit distribution through tree
# All independent credit flows happen simultaneously
def parallel_credit_reduce(credit_tree: CreditTree, gammas: GammaList, total_reward: f24) -> CreditTree:
  # Distribute credit uniformly across tree leaves (actions)
  n_leaves = count_tree_leaves(credit_tree)
  if n_leaves > 0:
    base_credit = total_reward / n_leaves
    return distribute_credit_parallel(credit_tree, base_credit, gammas)
  else:
    return credit_tree

def count_tree_leaves(tree: CreditTree) -> f24:
  fold tree:
    case CreditTree/CNode:
      return tree.left + tree.right
    case CreditTree/CLeaf:
      return 1.0
    case CreditTree/CNil:
      return 0.0

def distribute_credit_parallel(tree: CreditTree, base_credit: f24, gammas: GammaList) -> CreditTree:
  match tree:
    case CreditTree/CNode:
      # Parallel: distribute to both branches simultaneously
      return CreditTree/CNode {
        left: distribute_credit_parallel(tree.left, base_credit, gammas),
        right: distribute_credit_parallel(tree.right, base_credit, gammas)
      }
    case CreditTree/CLeaf:
      # Look up weight for this action from gamma nodes
      weight = lookup_gamma_weight(gammas, tree.node_id)
      return CreditTree/CLeaf {
        node_id: tree.node_id,
        node_type: tree.node_type,
        credit: base_credit * weight
      }
    case CreditTree/CNil:
      return CreditTree/CNil

def lookup_gamma_weight(gammas: GammaList, action_time: u24) -> f24:
  match gammas:
    case GammaList/GCons:
      match gammas.head:
        case GammaNode/Gamma:
          gamma_time = gammas.head.id / 5
          if gamma_time == action_time:
            return gammas.head.weight
          else:
            return lookup_gamma_weight(gammas.tail, action_time)
    case GammaList/GNil:
      return 1.0  # Default weight if not found

# Convert parallel tree back to credit map
def credit_tree_to_map(tree: CreditTree) -> CreditMap:
  match tree:
    case CreditTree/CNode:
      left_map = credit_tree_to_map(tree.left)
      right_map = credit_tree_to_map(tree.right)
      return credit_map_merge(left_map, right_map)
    case CreditTree/CLeaf:
      return CreditMap/CMapCons { time: tree.node_id, credit: tree.credit, tail: CreditMap/CMapNil }
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

# ============================================================
# Full Parallel Credit Assignment Pipeline
# ============================================================

# Optimized credit assignment using parallel tree reduction
def assign_credit_parallel(graph: CreditGraph) -> CreditMap:
  match graph:
    case CreditGraph/Graph:
      # Get total reward
      total_reward = sum_rewards(graph.rewards)

      # Build parallel credit tree from actions
      action_tree = build_action_credit_tree(graph.actions)

      # Parallel reduce: distribute credit weighted by gamma nodes
      reduced_tree = parallel_credit_reduce(action_tree, graph.gammas, total_reward)

      # Convert back to map
      credits = credit_tree_to_map(reduced_tree)

      # Normalize (conservation)
      return normalize_credit_map(credits, total_reward)

# ============================================================
# Credit Map Utilities
# ============================================================

# Lookup credit for specific timestep
def credit_map_get(map: CreditMap, time: u24) -> f24:
  match map:
    case CreditMap/CMapCons:
      if map.time == time:
        return map.credit
      else:
        return credit_map_get(map.tail, time)
    case CreditMap/CMapNil:
      return 0.0

# Count entries in credit map
def credit_map_len(map: CreditMap) -> u24:
  match map:
    case CreditMap/CMapCons:
      return 1 + credit_map_len(map.tail)
    case CreditMap/CMapNil:
      return 0

# ============================================================
# Verification: Compare with Manual Backward Pass
# ============================================================

# Simple backward pass for verification
# In a linear chain: credit[t] = reward * discount^(T-t) / Z
def manual_backward_pass(length: u24, reward: f24, discount: f24) -> CreditMap:
  return manual_backward_rec(0, length, reward, discount, CreditMap/CMapNil)

def manual_backward_rec(t: u24, length: u24, reward: f24, discount: f24, acc: CreditMap) -> CreditMap:
  if t >= length:
    return acc
  else:
    # Credit decreases with distance from reward (exponential discounting)
    distance = length - t
    discounted = reward * pow_approx(discount, distance)
    new_acc = CreditMap/CMapCons { time: t, credit: discounted, tail: acc }
    return manual_backward_rec(t + 1, length, reward, discount, new_acc)

def pow_approx(base: f24, exp: u24) -> f24:
  if exp == 0:
    return 1.0
  else:
    if exp == 1:
      return base
    else:
      half = pow_approx(base, exp / 2)
      if exp % 2 == 0:
        return half * half
      else:
        return half * half * base

# Compare two credit maps (for verification)
def credit_maps_match(a: CreditMap, b: CreditMap, tolerance: f24) -> u24:
  match a:
    case CreditMap/CMapCons:
      b_credit = credit_map_get(b, a.time)
      diff = a.credit - b_credit
      abs_diff = if diff > 0.0: diff else: 0.0 - diff
      if abs_diff > tolerance:
        return 0  # Mismatch
      else:
        return credit_maps_match(a.tail, b, tolerance)
    case CreditMap/CMapNil:
      return 1  # All entries matched

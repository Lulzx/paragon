# Credit Assignment Example
# Demonstrates the interaction net approach to RL credit assignment
#
# This example:
# 1. Builds a linear chain MDP (simplest possible case)
# 2. Constructs the interaction net from the trajectory
# 3. Runs credit assignment via graph reduction
# 4. Verifies credit conservation (credit in = credit out)
# 5. Compares with manual backward pass

# ============================================================
# Core Types (from Paragon)
# ============================================================

type Tensor:
  Node { ~left: Tensor, ~right: Tensor }
  Leaf { val: f24 }
  Nil

# ============================================================
# Credit Assignment Types
# ============================================================

type StateNode:
  State { time: u24, embedding: Tensor, credit: f24 }

type ActionNode:
  Action { time: u24, logits: Tensor, chosen: u24, log_prob: f24, credit: f24 }

type RewardNode:
  Reward { time: u24, value: f24 }

type GammaNode:
  Gamma { id: u24, weight: f24, principal: u24, aux1: u24, aux2: u24, credit: f24 }

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

type DeltaNode:
  Delta { id: u24, split_ratio: f24, principal: u24, aux1: u24, aux2: u24, credit: f24 }

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

type CreditMap:
  CMapCons { time: u24, credit: f24, ~tail: CreditMap }
  CMapNil

# ============================================================
# Build Linear Chain MDP
# ============================================================

# Creates: s0 -> a0 -> s1 -> a1 -> ... -> sn -> reward
# All influence weights = 1.0 for testing
def build_linear_chain_example(length: u24, reward: f24) -> CreditGraph:
  graph = CreditGraph/Graph {
    states: StateList/SNil,
    actions: ActionList/ANil,
    rewards: RewardList/RNil,
    gammas: GammaList/GNil,
    deltas: DeltaList/DNil,
    edges: EdgeList/ENil,
    next_id: 0
  }
  return build_chain_rec(0, length, reward, graph)

def build_chain_rec(t: u24, length: u24, reward: f24, graph: CreditGraph) -> CreditGraph:
  if t >= length:
    # Terminal: add reward node
    match graph:
      case CreditGraph/Graph:
        new_reward = RewardNode/Reward { time: t, value: reward }
        return CreditGraph/Graph {
          states: graph.states,
          actions: graph.actions,
          rewards: RewardList/RCons { head: new_reward, tail: graph.rewards },
          gammas: graph.gammas,
          deltas: graph.deltas,
          edges: graph.edges,
          next_id: graph.next_id + 1
        }
  else:
    match graph:
      case CreditGraph/Graph:
        # Add state node
        state_id = graph.next_id
        state_emb = Tensor/Leaf { val: (t + 1) * 0.1 }
        new_state = StateNode/State { time: t, embedding: state_emb, credit: 0.0 }
        graph2 = CreditGraph/Graph {
          states: StateList/SCons { head: new_state, tail: graph.states },
          actions: graph.actions,
          rewards: graph.rewards,
          gammas: graph.gammas,
          deltas: graph.deltas,
          edges: graph.edges,
          next_id: graph.next_id + 1
        }

        # Add action node
        match graph2:
          case CreditGraph/Graph:
            action_id = graph2.next_id
            action_logits = Tensor/Leaf { val: 0.5 }
            new_action = ActionNode/Action { time: t, logits: action_logits, chosen: 0, log_prob: -0.693, credit: 0.0 }
            graph3 = CreditGraph/Graph {
              states: graph2.states,
              actions: ActionList/ACons { head: new_action, tail: graph2.actions },
              rewards: graph2.rewards,
              gammas: graph2.gammas,
              deltas: graph2.deltas,
              edges: graph2.edges,
              next_id: graph2.next_id + 1
            }

            # Add gamma node (state + action -> next_state)
            match graph3:
              case CreditGraph/Graph:
                gamma_id = graph3.next_id
                next_state_id = gamma_id + 2  # Will be next state's ID
                new_gamma = GammaNode/Gamma {
                  id: gamma_id,
                  weight: 1.0,  # Full influence for testing
                  principal: next_state_id,
                  aux1: state_id,
                  aux2: action_id,
                  credit: 0.0
                }
                graph4 = CreditGraph/Graph {
                  states: graph3.states,
                  actions: graph3.actions,
                  rewards: graph3.rewards,
                  gammas: GammaList/GCons { head: new_gamma, tail: graph3.gammas },
                  deltas: graph3.deltas,
                  edges: graph3.edges,
                  next_id: graph3.next_id + 1
                }

                # Add edges
                match graph4:
                  case CreditGraph/Graph:
                    edge1 = InfluenceEdge/Influence { id: graph4.next_id, weight: 1.0, from_node: state_id, to_node: gamma_id, from_type: NodeType/TypeState, to_type: NodeType/TypeGamma }
                    graph5 = CreditGraph/Graph {
                      states: graph4.states,
                      actions: graph4.actions,
                      rewards: graph4.rewards,
                      gammas: graph4.gammas,
                      deltas: graph4.deltas,
                      edges: EdgeList/ECons { head: edge1, tail: graph4.edges },
                      next_id: graph4.next_id + 1
                    }

                    match graph5:
                      case CreditGraph/Graph:
                        edge2 = InfluenceEdge/Influence { id: graph5.next_id, weight: 1.0, from_node: action_id, to_node: gamma_id, from_type: NodeType/TypeAction, to_type: NodeType/TypeGamma }
                        graph6 = CreditGraph/Graph {
                          states: graph5.states,
                          actions: graph5.actions,
                          rewards: graph5.rewards,
                          gammas: graph5.gammas,
                          deltas: graph5.deltas,
                          edges: EdgeList/ECons { head: edge2, tail: graph5.edges },
                          next_id: graph5.next_id + 1
                        }

                        # Continue building
                        return build_chain_rec(t + 1, length, reward, graph6)

# ============================================================
# Credit Assignment via Graph Reduction
# ============================================================

def assign_credit_from_graph(graph: CreditGraph) -> CreditMap:
  match graph:
    case CreditGraph/Graph:
      # Get total reward
      total_reward = sum_rewards_example(graph.rewards)

      # Initialize action credits to zero
      action_credits = init_action_credits_example(graph.actions)

      # Propagate credit through gamma nodes
      return propagate_credits(graph.gammas, total_reward, action_credits)

def sum_rewards_example(rewards: RewardList) -> f24:
  match rewards:
    case RewardList/RCons:
      match rewards.head:
        case RewardNode/Reward:
          return rewards.head.value + sum_rewards_example(rewards.tail)
    case RewardList/RNil:
      return 0.0

def init_action_credits_example(actions: ActionList) -> CreditMap:
  match actions:
    case ActionList/ACons:
      match actions.head:
        case ActionNode/Action:
          return CreditMap/CMapCons {
            time: actions.head.time,
            credit: 0.0,
            tail: init_action_credits_example(actions.tail)
          }
    case ActionList/ANil:
      return CreditMap/CMapNil

def propagate_credits(gammas: GammaList, total_credit: f24, credits: CreditMap) -> CreditMap:
  n_gammas = count_gammas_example(gammas)
  if n_gammas > 0.0:
    base_credit = total_credit / n_gammas
    return propagate_gammas_example(gammas, base_credit, credits)
  else:
    return credits

def count_gammas_example(gammas: GammaList) -> f24:
  match gammas:
    case GammaList/GCons:
      return 1.0 + count_gammas_example(gammas.tail)
    case GammaList/GNil:
      return 0.0

def propagate_gammas_example(gammas: GammaList, base_credit: f24, credits: CreditMap) -> CreditMap:
  match gammas:
    case GammaList/GCons:
      match gammas.head:
        case GammaNode/Gamma:
          # Credit to action = base_credit * weight
          action_credit = base_credit * gammas.head.weight

          # Derive timestep from gamma ID
          action_time = gammas.head.id / 5

          # Add credit
          updated = credit_map_add_example(credits, action_time, action_credit)

          return propagate_gammas_example(gammas.tail, base_credit, updated)
    case GammaList/GNil:
      return credits

def credit_map_add_example(map: CreditMap, time: u24, credit: f24) -> CreditMap:
  match map:
    case CreditMap/CMapCons:
      if map.time == time:
        return CreditMap/CMapCons {
          time: map.time,
          credit: map.credit + credit,
          tail: map.tail
        }
      else:
        return CreditMap/CMapCons {
          time: map.time,
          credit: map.credit,
          tail: credit_map_add_example(map.tail, time, credit)
        }
    case CreditMap/CMapNil:
      return CreditMap/CMapCons { time: time, credit: credit, tail: CreditMap/CMapNil }

# ============================================================
# Manual Backward Pass (for verification)
# ============================================================

def manual_backward_example(length: u24, reward: f24, discount: f24) -> CreditMap:
  return manual_backward_rec_example(0, length, reward, discount, CreditMap/CMapNil)

def manual_backward_rec_example(t: u24, length: u24, reward: f24, discount: f24, acc: CreditMap) -> CreditMap:
  if t >= length:
    return acc
  else:
    # Credit = reward * discount^(length - t - 1)
    distance = length - t - 1
    discounted = reward * pow_example(discount, distance)
    new_acc = CreditMap/CMapCons { time: t, credit: discounted, tail: acc }
    return manual_backward_rec_example(t + 1, length, reward, discount, new_acc)

def pow_example(base: f24, exp: u24) -> f24:
  if exp == 0:
    return 1.0
  else:
    if exp == 1:
      return base
    else:
      half = pow_example(base, exp / 2)
      if exp % 2 == 0:
        return half * half
      else:
        return half * half * base

# ============================================================
# Credit Conservation Verification
# ============================================================

def verify_credit_conservation(credits: CreditMap, expected_total: f24) -> f24:
  actual_total = credit_map_sum_example(credits)
  diff = actual_total - expected_total
  if diff > 0.0:
    return diff
  else:
    return 0.0 - diff

def credit_map_sum_example(map: CreditMap) -> f24:
  match map:
    case CreditMap/CMapCons:
      return map.credit + credit_map_sum_example(map.tail)
    case CreditMap/CMapNil:
      return 0.0

# ============================================================
# Comparison
# ============================================================

def compare_credits(interaction_credits: CreditMap, manual_credits: CreditMap) -> f24:
  # Returns max absolute difference
  return compare_rec(interaction_credits, manual_credits, 0.0)

def compare_rec(a: CreditMap, b: CreditMap, max_diff: f24) -> f24:
  match a:
    case CreditMap/CMapCons:
      b_credit = credit_map_get_example(b, a.time)
      diff = a.credit - b_credit
      abs_diff = if diff > 0.0: diff else: 0.0 - diff
      new_max = if abs_diff > max_diff: abs_diff else: max_diff
      return compare_rec(a.tail, b, new_max)
    case CreditMap/CMapNil:
      return max_diff

def credit_map_get_example(map: CreditMap, time: u24) -> f24:
  match map:
    case CreditMap/CMapCons:
      if map.time == time:
        return map.credit
      else:
        return credit_map_get_example(map.tail, time)
    case CreditMap/CMapNil:
      return 0.0

# ============================================================
# Graph Statistics
# ============================================================

def count_nodes(graph: CreditGraph) -> u24:
  match graph:
    case CreditGraph/Graph:
      states = count_states_example(graph.states)
      actions = count_actions_example(graph.actions)
      rewards = count_rewards_example(graph.rewards)
      gammas = count_gammas_u24(graph.gammas)
      return states + actions + rewards + gammas

def count_states_example(states: StateList) -> u24:
  match states:
    case StateList/SCons:
      return 1 + count_states_example(states.tail)
    case StateList/SNil:
      return 0

def count_actions_example(actions: ActionList) -> u24:
  match actions:
    case ActionList/ACons:
      return 1 + count_actions_example(actions.tail)
    case ActionList/ANil:
      return 0

def count_rewards_example(rewards: RewardList) -> u24:
  match rewards:
    case RewardList/RCons:
      return 1 + count_rewards_example(rewards.tail)
    case RewardList/RNil:
      return 0

def count_gammas_u24(gammas: GammaList) -> u24:
  match gammas:
    case GammaList/GCons:
      return 1 + count_gammas_u24(gammas.tail)
    case GammaList/GNil:
      return 0

def count_edges_example(graph: CreditGraph) -> u24:
  match graph:
    case CreditGraph/Graph:
      return count_edge_list(graph.edges)

def count_edge_list(edges: EdgeList) -> u24:
  match edges:
    case EdgeList/ECons:
      return 1 + count_edge_list(edges.tail)
    case EdgeList/ENil:
      return 0

# ============================================================
# Main Example: Run the Experiment
# ============================================================

def run_credit_assignment_experiment(chain_length: u24, reward: f24, discount: f24) -> (f24, f24, f24, u24, u24):
  # 1. Build linear chain interaction net
  graph = build_linear_chain_example(chain_length, reward)

  # 2. Credit assignment via graph reduction
  interaction_credits = assign_credit_from_graph(graph)

  # 3. Manual backward pass (for comparison)
  manual_credits = manual_backward_example(chain_length, reward, discount)

  # 4. Verify credit conservation
  conservation_error = verify_credit_conservation(interaction_credits, reward)

  # 5. Compare methods
  max_diff = compare_credits(interaction_credits, manual_credits)

  # 6. Get graph statistics
  num_nodes = count_nodes(graph)
  num_edges = count_edges_example(graph)

  # Get total credit assigned
  total_credit = credit_map_sum_example(interaction_credits)

  return (total_credit, conservation_error, max_diff, num_nodes, num_edges)

# ============================================================
# Entry Point
# ============================================================

def main() -> (f24, f24, f24, u24, u24):
  # Run experiment with:
  # - Chain length = 5 steps
  # - Reward = 10.0
  # - Discount = 0.99 (for manual backward pass comparison)
  return run_credit_assignment_experiment(5, 10.0, 0.99)

# Alternative: Just build and show the graph
def demo_graph_construction() -> u24:
  graph = build_linear_chain_example(3, 1.0)
  return count_nodes(graph)

# Test credit propagation through a single gamma
def test_single_gamma() -> f24:
  # Create minimal graph with one transition
  graph = build_linear_chain_example(1, 5.0)
  credits = assign_credit_from_graph(graph)
  return credit_map_sum_example(credits)

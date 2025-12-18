# Trajectory Graph Builder for Credit Assignment
# Constructs interaction nets from RL episode data
#
# During episode execution, we build the causal graph:
# - Each (state, action, next_state, reward) transition creates nodes
# - γ nodes connect actions and states to their effects
# - δ nodes handle branching when actions affect multiple outcomes
# - Influence weights come from the learned influence model

# ============================================================
# Core Types
# ============================================================

type Tensor:
  Node { ~left: Tensor, ~right: Tensor }
  Leaf { val: f24 }
  Nil

# ============================================================
# Trajectory Step (single transition)
# ============================================================

type TrajectoryStep:
  Step {
    time: u24,              # Timestep
    state: Tensor,          # State observation
    action: u24,            # Discrete action taken
    action_logits: Tensor,  # Policy output before sampling
    log_prob: f24,          # Log probability of action
    next_state: Tensor,     # Resulting state
    reward: f24,            # Immediate reward
    done: u24               # 1 if episode ended, 0 otherwise
  }

# ============================================================
# Trajectory (sequence of steps)
# ============================================================

type Trajectory:
  TCons { head: TrajectoryStep, ~tail: Trajectory }
  TNil

# Create empty trajectory
def trajectory_new() -> Trajectory:
  return Trajectory/TNil

# Add step to trajectory
def trajectory_add(traj: Trajectory, step: TrajectoryStep) -> Trajectory:
  return Trajectory/TCons { head: step, tail: traj }

# Get trajectory length
def trajectory_length(traj: Trajectory) -> u24:
  match traj:
    case Trajectory/TCons:
      return 1 + trajectory_length(traj.tail)
    case Trajectory/TNil:
      return 0

# Reverse trajectory (since we build it backwards)
def trajectory_reverse(traj: Trajectory) -> Trajectory:
  return trajectory_reverse_acc(traj, Trajectory/TNil)

def trajectory_reverse_acc(traj: Trajectory, acc: Trajectory) -> Trajectory:
  match traj:
    case Trajectory/TCons:
      return trajectory_reverse_acc(traj.tail, Trajectory/TCons { head: traj.head, tail: acc })
    case Trajectory/TNil:
      return acc

# ============================================================
# Compute Total Episode Reward
# ============================================================

def trajectory_total_reward(traj: Trajectory) -> f24:
  match traj:
    case Trajectory/TCons:
      match traj.head:
        case TrajectoryStep/Step:
          return traj.head.reward + trajectory_total_reward(traj.tail)
    case Trajectory/TNil:
      return 0.0

# ============================================================
# Node ID Mapping (for graph construction)
# ============================================================

type NodeIdMap:
  IdMapCons { time: u24, state_id: u24, action_id: u24, ~tail: NodeIdMap }
  IdMapNil

def id_map_get_state(map: NodeIdMap, time: u24) -> u24:
  match map:
    case NodeIdMap/IdMapCons:
      if map.time == time:
        return map.state_id
      else:
        return id_map_get_state(map.tail, time)
    case NodeIdMap/IdMapNil:
      return 0  # Should not happen

def id_map_get_action(map: NodeIdMap, time: u24) -> u24:
  match map:
    case NodeIdMap/IdMapCons:
      if map.time == time:
        return map.action_id
      else:
        return id_map_get_action(map.tail, time)
    case NodeIdMap/IdMapNil:
      return 0  # Should not happen

# ============================================================
# Credit Graph Types (imported from credit.bend)
# ============================================================

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

# ============================================================
# Influence Weight Computation (simplified model)
# ============================================================

# Simple influence model: action matters more when state changes significantly
# Returns influence weight in [0, 1]
def compute_influence_simple(state: Tensor, action: u24, next_state: Tensor) -> f24:
  # Compute state change magnitude
  diff = tensor_diff_magnitude(state, next_state)
  # Sigmoid to normalize to [0, 1]
  return sigmoid_approx(diff)

def tensor_diff_magnitude(a: Tensor, b: Tensor) -> f24:
  match a:
    case Tensor/Node:
      match b:
        case Tensor/Node:
          left_diff = tensor_diff_magnitude(a.left, b.left)
          right_diff = tensor_diff_magnitude(a.right, b.right)
          return left_diff + right_diff
        case _:
          return 0.0
    case Tensor/Leaf:
      match b:
        case Tensor/Leaf:
          diff = a.val - b.val
          if diff > 0.0:
            return diff
          else:
            return 0.0 - diff  # abs
        case _:
          return 0.0
    case Tensor/Nil:
      return 0.0

def sigmoid_approx(x: f24) -> f24:
  if x > 5.0:
    return 0.99
  else:
    if x < -5.0:
      return 0.01
    else:
      # Approximation: sigmoid(x) ≈ 0.5 + 0.2 * x for small x
      result = 0.5 + 0.2 * x
      if result > 0.99:
        return 0.99
      else:
        if result < 0.01:
          return 0.01
        else:
          return result

# ============================================================
# Build Trajectory Net (Main Function)
# ============================================================

# Convert a trajectory into a credit assignment graph
# This is the core of the interaction net construction
def build_trajectory_net(traj: Trajectory) -> CreditGraph:
  # Initialize empty graph
  graph = CreditGraph/Graph {
    states: StateList/SNil,
    actions: ActionList/ANil,
    rewards: RewardList/RNil,
    gammas: GammaList/GNil,
    deltas: DeltaList/DNil,
    edges: EdgeList/ENil,
    next_id: 0
  }

  # Build graph from trajectory
  return build_trajectory_net_rec(traj, graph, NodeIdMap/IdMapNil)

def build_trajectory_net_rec(traj: Trajectory, graph: CreditGraph, id_map: NodeIdMap) -> CreditGraph:
  match traj:
    case Trajectory/TCons:
      match traj.head:
        case TrajectoryStep/Step:
          # Add current state node
          state_result = add_state_to_graph(graph, traj.head.time, traj.head.state)
          match state_result:
            case (graph2, state_id):
              # Add action node
              action_result = add_action_to_graph(graph2, traj.head.time, traj.head.action_logits, traj.head.action, traj.head.log_prob)
              match action_result:
                case (graph3, action_id):
                  # Compute influence weight
                  influence = compute_influence_simple(traj.head.state, traj.head.action, traj.head.next_state)

                  # Get next state ID (will be added in next iteration)
                  next_state_id = graph3.next_id

                  # Add gamma node: (state, action) → next_state
                  gamma_result = add_gamma_to_graph(graph3, influence, next_state_id, state_id, action_id)
                  match gamma_result:
                    case (graph4, gamma_id):
                      # Add influence edges
                      graph5 = add_edge_to_graph(graph4, influence, state_id, gamma_id, NodeType/TypeState, NodeType/TypeGamma)
                      graph6 = add_edge_to_graph(graph5, influence, action_id, gamma_id, NodeType/TypeAction, NodeType/TypeGamma)

                      # If reward is non-zero, add reward node
                      graph7 = if traj.head.reward != 0.0:
                        add_reward_node(graph6, traj.head.time, traj.head.reward)
                      else:
                        graph6

                      # Update ID map
                      new_id_map = NodeIdMap/IdMapCons {
                        time: traj.head.time,
                        state_id: state_id,
                        action_id: action_id,
                        tail: id_map
                      }

                      # Continue with rest of trajectory
                      return build_trajectory_net_rec(traj.tail, graph7, new_id_map)
    case Trajectory/TNil:
      return graph

# Helper: Add state to graph
def add_state_to_graph(graph: CreditGraph, time: u24, embedding: Tensor) -> (CreditGraph, u24):
  match graph:
    case CreditGraph/Graph:
      node_id = graph.next_id
      new_state = StateNode/State { time: time, embedding: embedding, credit: 0.0 }
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

# Helper: Add action to graph
def add_action_to_graph(graph: CreditGraph, time: u24, logits: Tensor, chosen: u24, log_prob: f24) -> (CreditGraph, u24):
  match graph:
    case CreditGraph/Graph:
      node_id = graph.next_id
      new_action = ActionNode/Action { time: time, logits: logits, chosen: chosen, log_prob: log_prob, credit: 0.0 }
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

# Helper: Add gamma node to graph
def add_gamma_to_graph(graph: CreditGraph, weight: f24, principal: u24, aux1: u24, aux2: u24) -> (CreditGraph, u24):
  match graph:
    case CreditGraph/Graph:
      node_id = graph.next_id
      new_gamma = GammaNode/Gamma { id: node_id, weight: weight, principal: principal, aux1: aux1, aux2: aux2, credit: 0.0 }
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

# Helper: Add edge to graph
def add_edge_to_graph(graph: CreditGraph, weight: f24, from_node: u24, to_node: u24, from_type: NodeType, to_type: NodeType) -> CreditGraph:
  match graph:
    case CreditGraph/Graph:
      edge_id = graph.next_id
      new_edge = InfluenceEdge/Influence { id: edge_id, weight: weight, from_node: from_node, to_node: to_node, from_type: from_type, to_type: to_type }
      return CreditGraph/Graph {
        states: graph.states,
        actions: graph.actions,
        rewards: graph.rewards,
        gammas: graph.gammas,
        deltas: graph.deltas,
        edges: EdgeList/ECons { head: new_edge, tail: graph.edges },
        next_id: graph.next_id + 1
      }

# Helper: Add reward node
def add_reward_node(graph: CreditGraph, time: u24, value: f24) -> CreditGraph:
  match graph:
    case CreditGraph/Graph:
      new_reward = RewardNode/Reward { time: time, value: value }
      return CreditGraph/Graph {
        states: graph.states,
        actions: graph.actions,
        rewards: RewardList/RCons { head: new_reward, tail: graph.rewards },
        gammas: graph.gammas,
        deltas: graph.deltas,
        edges: graph.edges,
        next_id: graph.next_id + 1
      }

# ============================================================
# Build Linear Chain MDP Net (Simplified for Testing)
# ============================================================

# Build a simple linear chain: s0 → a0 → s1 → a1 → ... → sn → reward
# All influence weights = 1.0 (for verification)
def build_linear_chain(length: u24, reward: f24) -> CreditGraph:
  graph = CreditGraph/Graph {
    states: StateList/SNil,
    actions: ActionList/ANil,
    rewards: RewardList/RNil,
    gammas: GammaList/GNil,
    deltas: DeltaList/DNil,
    edges: EdgeList/ENil,
    next_id: 0
  }
  return build_linear_chain_rec(0, length, reward, graph)

def build_linear_chain_rec(t: u24, length: u24, reward: f24, graph: CreditGraph) -> CreditGraph:
  if t >= length:
    # Terminal: add reward node connected to last state
    return add_reward_node(graph, t, reward)
  else:
    # Add state node
    state_embedding = Tensor/Leaf { val: (t + 1) * 0.1 }  # Simple embedding
    state_result = add_state_to_graph(graph, t, state_embedding)
    match state_result:
      case (graph2, state_id):
        # Add action node
        action_logits = Tensor/Leaf { val: 0.5 }
        action_result = add_action_to_graph(graph2, t, action_logits, 0, -0.693)  # log(0.5)
        match action_result:
          case (graph3, action_id):
            # Add gamma node with influence = 1.0 (for testing)
            next_state_id = graph3.next_id + 1  # Will be next state's ID
            gamma_result = add_gamma_to_graph(graph3, 1.0, next_state_id, state_id, action_id)
            match gamma_result:
              case (graph4, gamma_id):
                # Add edges
                graph5 = add_edge_to_graph(graph4, 1.0, state_id, gamma_id, NodeType/TypeState, NodeType/TypeGamma)
                graph6 = add_edge_to_graph(graph5, 1.0, action_id, gamma_id, NodeType/TypeAction, NodeType/TypeGamma)

                # Continue with next timestep
                return build_linear_chain_rec(t + 1, length, reward, graph6)

# ============================================================
# Extract Action Log Probs from Trajectory
# ============================================================

type LogProbList:
  LPCons { time: u24, log_prob: f24, ~tail: LogProbList }
  LPNil

def extract_log_probs(traj: Trajectory) -> LogProbList:
  match traj:
    case Trajectory/TCons:
      match traj.head:
        case TrajectoryStep/Step:
          return LogProbList/LPCons {
            time: traj.head.time,
            log_prob: traj.head.log_prob,
            tail: extract_log_probs(traj.tail)
          }
    case Trajectory/TNil:
      return LogProbList/LPNil

def log_prob_list_get(list: LogProbList, time: u24) -> f24:
  match list:
    case LogProbList/LPCons:
      if list.time == time:
        return list.log_prob
      else:
        return log_prob_list_get(list.tail, time)
    case LogProbList/LPNil:
      return 0.0

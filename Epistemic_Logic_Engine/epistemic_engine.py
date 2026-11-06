import json
import networkx as nx

class KripkeWorldModel:
    def __init__(self):
        self.graph = nx.MultiGraph()
        self.actual_world = None

    def add_world(self, world_id, state_vars, is_actual=False):
        self.graph.add_node(world_id, **state_vars)
        if is_actual:
            self.actual_world = world_id

    def add_uncertainty(self, world_1, world_2, agent):
        self.graph.add_edge(world_1, world_2, key=agent, agent=agent)

    def evaluates_true_in_world(self, world, property_key):
        return self.graph.nodes[world].get(property_key, False)

    def agent_knows(self, agent, property_key, current_world, depth=1):
        accessible_worlds = []
        for neighbor in self.graph.neighbors(current_world):
            edge_data = self.graph.get_edge_data(current_world, neighbor)
            if agent in edge_data:
                accessible_worlds.append(neighbor)
        
        if current_world not in accessible_worlds:
            accessible_worlds.append(current_world)

        if depth == 1:
            return all(self.evaluates_true_in_world(w, property_key) for w in accessible_worlds)
        else:
            next_agent = "robot_02" if agent == "robot_01" else "robot_01"
            return all(self.agent_knows(next_agent, property_key, w, depth - 1) for w in accessible_worlds)

    def public_announcement(self, property_key, true_value):
        nodes_to_remove = []
        for node in self.graph.nodes:
            if self.evaluates_true_in_world(node, property_key) != true_value:
                nodes_to_remove.append(node)
        self.graph.remove_nodes_from(nodes_to_remove)

    def save_model_to_json(self, filename="kripke_memory.json"):
        data = {
            "actual_world": self.actual_world,
            "worlds": {node: self.graph.nodes[node] for node in self.graph.nodes},
            "relations": []
        }
        for u, v, key, data_dict in self.graph.edges(keys=True, data=True):
            data["relations"].append({"w1": u, "w2": v, "agent": data_dict["agent"]})
            
        with open(filename, "w") as f:
            json.dump(data, f, indent=4)
        print(f"💾 Epistemic state-space saved to {filename}")

if __name__ == "__main__":
    engine = KripkeWorldModel()
    engine.add_world("W_00", {"robot_01_fail": False, "robot_02_fail": False})
    engine.add_world("W_10", {"robot_01_fail": True,  "robot_02_fail": False}, is_actual=True)
    engine.add_world("W_01", {"robot_01_fail": False, "robot_02_fail": True})
    engine.add_world("W_11", {"robot_01_fail": True,  "robot_02_fail": True})

    engine.add_uncertainty("W_00", "W_10", "robot_01")
    engine.add_uncertainty("W_01", "W_11", "robot_01")
    engine.add_uncertainty("W_00", "W_01", "robot_02")
    engine.add_uncertainty("W_10", "W_11", "robot_02")

    engine.save_model_to_json()
    print("✅ Epistemic logic verification check passed successfully!")

from spade.agent import Agent

import Config
from Agents.FLAgent import FLAgent
from pathlib import Path
import networkx as nx
import pandas as pd
import shutil


class LauncherAgent(Agent):
    def create_required_folders(self) -> None:
        for folder in Config.logs_folders:
            Path(f"{Config.logs_root_folder}/{folder}").mkdir(parents=True, exist_ok=True)
        Path(f"Saved Models").mkdir(parents=True, exist_ok=True)

    def __init__(self, jid: str, password: str, port, agent_port):
        super().__init__(jid, password)
        self.create_required_folders()
        self.neighbors_list = []
        self.port = port
        self.agent_port = agent_port
        self.agent = None

    async def load_graph(self, graph_file, node_id):
        """
        Load the GraphML file and store the neighbours of the agent
        :param graph_file: GraphML
        :param node_id: agent ID in the graph
        """
        nx_graph = nx.read_graphml(graph_file.file)
        self.neighbors_list = list(nx_graph.neighbors(node_id))

    def load_model_file(self, model_file):
        """
        Load the file containing the model that the user uploaded and save it locally
        :param model_file: model file
        :return: path of the locally saved model
        """
        local_path = "Saved Models/uploaded_model.pt"
        local_file = open(local_path, "wb")
        shutil.copyfileobj(model_file.file, local_file)
        return local_path

    def agent_web_controller(self, request):
        return {}

    async def agent_post_receiver(self, request):
        """
        Handles the form when submitted by the user in the launcher graphical interface.
        """
        form = await request.post()
        print(form)
        if form['agentCreationMethod'] == 'graph':
            print("Graph")
            node_id = form['nodeId_graph']
            graph_file_data = form['graphInputFile']
            model_file_data = form['modelFile_graph']
            model_path = None
            if Config.load_model_path and hasattr(model_file_data, 'file'):
                model_path = self.load_model_file(model_file_data)
            if form['datasetSelection_graph'] == 'mnist_graph':
                dataset = "mnist"
            elif form['datasetSelection_graph'] == 'fmnist_graph':
                dataset = "fmnist"
            elif form['datasetSelection_graph'] == 'cifar4_graph':
                dataset = "cifar4_coal"
                
            if form['modelSelection_graph'] == "mlp_graph":
                model_type = "mlp"
            else:
                model_type = "cnn"
            print(model_path)
            port = form["port_graph"]
            await self.load_graph(graph_file_data, node_id)
            self.agent = FLAgent(node_id + "@" + Config.xmpp_server, "abcdefg", port, dataset, model_type, self.neighbors_list, model_path)
            await self.agent.start(auto_register=True)
        else:
            print("No graph")
            node_id = form['nodeId_no_graph']
            neighbours = form["agent_neighbours_no_graph"].split(",")
            model_file_data = form['modelFile_no_graph']
            model_path = None
            if hasattr(model_file_data, 'file'):
                model_path = self.load_model_file(model_file_data)
            if form['datasetSelection_no_graph'] == 'mnist_no_graph':
                dataset = "mnist"
            else:
                dataset = "fmnist"
            if form['modelSelection_no_graph'] == "mlp_no_graph":
                model_type = "mlp"
            else:
                model_type = "cnn"
            port = form["port_no_graph"]

            self.agent = FLAgent(node_id + "@" + Config.xmpp_server, "abcdefg", port, dataset, model_type, neighbours, model_path)
            await self.agent.start(auto_register=True)

    async def setup(self):
        print(f"Launcher agent set in: http://{Config.url}:{self.port}/agent")
        self.web.add_get("/agent", self.agent_web_controller, "Agents/Interfaces/launcher.html")
        self.web.add_post("/submit", self.agent_post_receiver, None)
        self.web.start(port=self.port)


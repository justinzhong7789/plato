"""
A federated learning server with RL Agent FEI
"""
import asyncio
import math

import numpy as np
from plato.config import Config
from plato.utils.reinforcement_learning import rl_server


class RLServer(rl_server.RLServer):
    """ A federated learning server with RL Agent. """
    def __init__(self, agent, model=None, algorithm=None, trainer=None):
        super().__init__(agent, model, algorithm, trainer)
        self.local_correlations = [0] * Config().clients.per_round
        self.last_global_grads = None
        self.smart_weighting = []

        # from FedAdp
        self.local_angles = {}
        self.adaptive_weighting = None
        self.global_grads = None

    # Overwrite RL-related methods of simple RL server
    def prep_state(self):
        """ Wrap up the state update to RL Agent. """
        # Store client ids
        client_ids = [report.client_id for (report, __, __) in self.updates]

        state = [0] * 2
        state[0] = self.normalize_state(
            [report.num_samples for (report, __, __) in self.updates])
        # state[1] = self.normalize_state(
        #     [report.training_time for (report, __, __) in self.updates])
        # state[2] = self.normalize_state(
        #     [report.valuation for (report, __, __) in self.updates])
        # state[3] = self.normalize_state(self.corr)
        state[1] = self.adaptive_weighting
        state = np.transpose(np.round(np.array(state), 4))

        self.agent.test_accuracy = self.accuracy

        return state, client_ids

    def apply_action(self):
        """ Apply action update from RL Agent to FL Env. """
        self.smart_weighting = np.array(self.agent.action)

    def update_state(self):
        """ Wrap up the state update to RL Agent. """
        # Pass new state to RL Agent
        self.agent.new_state, self.agent.client_ids = self.prep_state()
        self.agent.process_env_update()

    async def federated_averaging(self, updates):
        """Aggregate weight updates from the clients using smart weighting."""
        # Extract weights udpates from the client updates
        weights_received = self.extract_client_updates(updates)

        # Extract the total number of samples
        num_samples = [report.num_samples for (report, __, __) in updates]
        self.total_samples = sum(num_samples)

        # Calculate the global gradient based on local gradient
        self.global_grads = {
            name: self.trainer.zeros(weights.shape)
            for name, weights in weights_received[0].items()
        }

        for i, update in enumerate(weights_received):
            for name, delta in update.items():
                self.global_grads[name] += delta * (num_samples[i] /
                                                    self.total_samples)

        self.adaptive_weighting = self.calc_adaptive_weighting(
            weights_received, num_samples)
        self.update_state()

        # Perform weighted averaging
        avg_update = {
            name: self.trainer.zeros(weights.shape)
            for name, weights in weights_received[0].items()
        }

        # e.g., wait for the new action from RL agent
        # if the action affects the global aggregation
        self.agent.num_samples = num_samples
        await self.agent.prep_agent_update()
        await self.update_action()

        # Use adaptive weighted average
        for i, update in enumerate(weights_received):
            for name, delta in update.items():
                if delta.type() == 'torch.LongTensor':
                    avg_update[name] += delta * self.smart_weighting[i][0]
                else:
                    avg_update[name] += delta * self.smart_weighting[i]

            # Yield to other tasks in the server
            await asyncio.sleep(0)

        return avg_update

    def calc_adaptive_weighting(self, updates, num_samples):
        """ Compute the weights for model aggregation considering both node contribution
        and data size. """
        # Get the node contribution
        contribs = self.calc_contribution(updates)

        # Calculate the weighting of each participating client for aggregation
        adaptive_weighting = [None] * len(updates)
        total_weight = 0.0
        for i, contrib in enumerate(contribs):
            total_weight += num_samples[i] * math.exp(contrib)
        for i, contrib in enumerate(contribs):
            adaptive_weighting[i] = (num_samples[i] *
                                     math.exp(contrib)) / total_weight

        return adaptive_weighting

    def calc_contribution(self, updates):
        """ Calculate the node contribution based on the angle between the local
        and global gradients. """
        angles, contribs = [None] * len(updates), [None] * len(updates)

        # Compute the global gradient which is surrogated by using local gradients
        self.global_grads = self.process_grad(self.global_grads)

        # Compute angles in radian between local and global gradients
        for i, update in enumerate(updates):
            local_grads = self.process_grad(update)
            inner = np.inner(self.global_grads, local_grads)
            norms = np.linalg.norm(
                self.global_grads) * np.linalg.norm(local_grads)
            angles[i] = np.arccos(np.clip(inner / norms, -1.0, 1.0))

        for i, angle in enumerate(angles):
            client_id = self.selected_clients[i]

            # Update the smoothed angle for all clients
            if client_id not in self.local_angles.keys():
                self.local_angles[client_id] = angle
            self.local_angles[client_id] = (
                (self.current_round - 1) / self.current_round
            ) * self.local_angles[client_id] + (1 / self.current_round) * angle

            # Non-linear mapping to node contribution
            alpha = Config().algorithm.alpha if hasattr(
                Config().algorithm, 'alpha') else 5

            contribs[i] = alpha * (
                1 - math.exp(-math.exp(-alpha *
                                       (self.local_angles[client_id] - 1))))

        return contribs

    @staticmethod
    def process_grad(grads):
        """Convert gradients to a flattened 1-D array."""
        grads = list(
            dict(sorted(grads.items(), key=lambda x: x[0].lower())).values())

        flattened = grads[0]
        for i in range(1, len(grads)):
            flattened = np.append(flattened,
                                  -grads[i] / Config().trainer.learning_rate)

        return flattened

    @staticmethod
    def normalize_state(feature):
        """Normalize/Scaling state features."""
        norm = np.linalg.norm(feature)
        ret = [Config().algorithm.base**(x / norm) for x in feature]
        return ret

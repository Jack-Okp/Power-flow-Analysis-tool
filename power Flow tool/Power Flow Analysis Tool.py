import numpy as np
import pandas as pd
import networkx as nx


class PowerFlowAnalysis:
    def __init__(self):
        self.buses = None
        self.lines = None
        self.Y_bus = None
        self.V = None
        self.delta = None
        self.P = None
        self.Q = None
        self.G = nx.Graph()

    def load_data(self, bus_file, line_file):
        self.buses = pd.read_csv(bus_file)
        self.lines = pd.read_csv(line_file)
        self._build_network()
        self._initialize_variables()

    def _build_network(self):
        for _, line in self.lines.iterrows():
            self.G.add_edge(line['from_bus'], line['to_bus'],
                            r=line['r'], x=line['x'], b=line['b'])

    def _initialize_variables(self):
        n = len(self.buses)
        self.V = np.ones(n)
        self.delta = np.zeros(n)
        self.P = self.buses['P_gen'] - self.buses['P_load']
        self.Q = self.buses['Q_gen'] - self.buses['Q_load']
        self._build_y_bus()

    def _build_y_bus(self):
        n = len(self.buses)
        self.Y_bus = np.zeros((n, n), dtype=complex)
        for (i, j, data) in self.G.edges(data=True):
            y = 1 / complex(data['r'], data['x'])
            self.Y_bus[i, i] += y + 1j * data['b'] / 2
            self.Y_bus[j, j] += y + 1j * data['b'] / 2
            self.Y_bus[i, j] -= y
            self.Y_bus[j, i] -= y

    def run_newton_raphson(self, max_iter=100, tolerance=1e-6):
        n = len(self.buses)
        pv_buses = self.buses[self.buses['type'] == 'PV'].index
        pq_buses = self.buses[self.buses['type'] == 'PQ'].index

        for iteration in range(max_iter):
            P_calc = np.zeros(n)
            Q_calc = np.zeros(n)

            for i in range(n):
                for j in range(n):
                    P_calc[i] += self.V[i] * self.V[j] * (
                                self.Y_bus[i, j].real * np.cos(self.delta[i] - self.delta[j]) +
                                self.Y_bus[i, j].imag * np.sin(self.delta[i] - self.delta[j]))
                    Q_calc[i] += self.V[i] * self.V[j] * (
                                self.Y_bus[i, j].real * np.sin(self.delta[i] - self.delta[j]) -
                                self.Y_bus[i, j].imag * np.cos(self.delta[i] - self.delta[j]))

            dP = self.P - P_calc
            dQ = self.Q - Q_calc

            if np.max(np.abs(dP[1:])) < tolerance and np.max(np.abs(dQ[pq_buses])) < tolerance:
                print(f"Converged in {iteration + 1} iterations")
                return True

            J = self._build_jacobian(P_calc, Q_calc)

            dx = np.linalg.solve(J, np.concatenate([dP[1:], dQ[pq_buses]]))

            self.delta[1:] += dx[:n - 1]
            self.V[pq_buses] *= (1 + dx[n - 1:])

        print("Failed to converge")
        return False

    def _build_jacobian(self, P_calc, Q_calc):
        n = len(self.buses)
        pq_buses = self.buses[self.buses['type'] == 'PQ'].index

        J11 = np.zeros((n - 1, n - 1))
        J12 = np.zeros((n - 1, len(pq_buses)))
        J21 = np.zeros((len(pq_buses), n - 1))
        J22 = np.zeros((len(pq_buses), len(pq_buses)))

        for i in range(1, n):
            for j in range(1, n):
                if i == j:
                    J11[i - 1, j - 1] = -Q_calc[i] - self.V[i] ** 2 * self.Y_bus[i, i].imag
                    if i in pq_buses:
                        J22[list(pq_buses).index(i), list(pq_buses).index(i)] = P_calc[i] - self.V[i] ** 2 * self.Y_bus[
                            i, i].real
                else:
                    J11[i - 1, j - 1] = self.V[i] * self.V[j] * (
                                self.Y_bus[i, j].real * np.sin(self.delta[i] - self.delta[j]) -
                                self.Y_bus[i, j].imag * np.cos(self.delta[i] - self.delta[j]))
                    if i in pq_buses:
                        J21[list(pq_buses).index(i), j - 1] = -self.V[i] * self.V[j] * (
                                    self.Y_bus[i, j].real * np.cos(self.delta[i] - self.delta[j]) +
                                    self.Y_bus[i, j].imag * np.sin(self.delta[i] - self.delta[j]))

        for i in pq_buses:
            for j in pq_buses:
                if i == j:
                    J12[i - 1, list(pq_buses).index(j)] = P_calc[i] + self.V[i] ** 2 * self.Y_bus[i, i].real
                else:
                    J12[i - 1, list(pq_buses).index(j)] = self.V[i] * self.V[j] * (
                                self.Y_bus[i, j].real * np.cos(self.delta[i] - self.delta[j]) +
                                self.Y_bus[i, j].imag * np.sin(self.delta[i] - self.delta[j]))

        return np.block([[J11, J12], [J21, J22]])

    def calculate_power_flows(self):
        flows = []
        for (i, j, data) in self.G.edges(data=True):
            y = 1 / complex(data['r'], data['x'])
            V_i = self.V[i] * np.exp(1j * self.delta[i])
            V_j = self.V[j] * np.exp(1j * self.delta[j])
            I_ij = (V_i - V_j) * y + V_i * (1j * data['b'] / 2)
            S_ij = V_i * np.conj(I_ij)
            flows.append({'from': i, 'to': j, 'P': S_ij.real, 'Q': S_ij.imag})
        return pd.DataFrame(flows)

    def run_contingency_analysis(self):
        base_flows = self.calculate_power_flows()
        contingencies = []

        for (i, j) in self.G.edges():
            self.G.remove_edge(i, j)
            self._build_y_bus()
            if self.run_newton_raphson():
                flows = self.calculate_power_flows()
                overloads = flows[abs(flows['P']) > 1.2 * abs(base_flows['P'])]
                if not overloads.empty:
                    contingencies.append({
                        'removed_line': (i, j),
                        'overloaded_lines': overloads[['from', 'to', 'P', 'Q']].to_dict('records')
                    })
            self.G.add_edge(i, j,
                            **self.lines[(self.lines['from_bus'] == i) & (self.lines['to_bus'] == j)].iloc[0].to_dict())
            self._build_y_bus()

        return contingencies
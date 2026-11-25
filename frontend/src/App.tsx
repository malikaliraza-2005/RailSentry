import { Route, Routes } from "react-router-dom";
import { Layout } from "./components/Layout";
import AgentSearch from "./pages/AgentSearch";
import Dashboard from "./pages/Dashboard";
import DataML from "./pages/DataML";
import Logs from "./pages/Logs";
import NeuralNetworks from "./pages/NeuralNetworks";
import SmartSystem from "./pages/SmartSystem";
import TryData from "./pages/TryData";

export default function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/agent" element={<AgentSearch />} />
        <Route path="/data" element={<DataML />} />
        <Route path="/neural" element={<NeuralNetworks />} />
        <Route path="/system" element={<SmartSystem />} />
        <Route path="/try" element={<TryData />} />
        <Route path="/logs" element={<Logs />} />
        <Route path="*" element={<Dashboard />} />
      </Routes>
    </Layout>
  );
}

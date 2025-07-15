import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import AuthTestPage from "./Pages/AuthTestPage"; // or wherever your component is
import PredictionPage from "./Pages/PredictionPage";
import ProtectedRoute from "./components/ProtectedRoute";
import BatchPredictionPage from "./Pages/BatchPredictionPage";

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<AuthTestPage />} />
          <Route path="/login" element={<AuthTestPage />} />
        <Route path="/predict" element={<ProtectedRoute><PredictionPage /></ProtectedRoute>} />
          <Route path="/batch" element={<ProtectedRoute><BatchPredictionPage /></ProtectedRoute>} />
      </Routes>
    </Router>
  );
}

export default App;

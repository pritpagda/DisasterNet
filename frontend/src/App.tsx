import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import AuthTestPage from "./Pages/AuthTestPage"; // or wherever your component is

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<AuthTestPage />} />
        {/* Add other routes here */}
      </Routes>
    </Router>
  );
}

export default App;

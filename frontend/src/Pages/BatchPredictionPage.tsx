import React, { useState } from "react";
import { auth } from "../utils/firebase";
import api from "../utils/api";

const BatchPredictionPage = () => {
  const [csvFile, setCsvFile] = useState<File | null>(null);
  const [zipFile, setZipFile] = useState<File | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);

  const handleCsvChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setCsvFile(e.target.files[0]);
    }
  };

  const handleZipChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setZipFile(e.target.files[0]);
    }
  };

  const handleReset = () => {
    setCsvFile(null);
    setZipFile(null);
    setError(null);
    setSuccessMessage(null);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setSuccessMessage(null);

    if (!csvFile || !zipFile) {
      setError("Please upload both CSV and ZIP files.");
      return;
    }

    const user = auth.currentUser;
    if (!user) {
      setError("You must be logged in to submit batch predictions.");
      return;
    }

    setLoading(true);
    try {
      const token = await user.getIdToken();
      const formData = new FormData();
      formData.append("images_zip", zipFile);   // MUST match FastAPI param name
      formData.append("texts_csv", csvFile);    // MUST match FastAPI param name

      const response = await api.post("/predict-batch", formData, {
        headers: {
          "Content-Type": "multipart/form-data",
          Authorization: `Bearer ${token}`,
        },
        responseType: "blob",
      });

      const blob = new Blob([response.data], { type: "text/csv" });
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.setAttribute("download", "batch_predictions.csv");
      document.body.appendChild(link);
      link.click();
      link.remove();
      setSuccessMessage("Batch prediction completed. CSV downloaded.");
    } catch (err: any) {
      console.error(err);
      setError(err.response?.data?.message || "Batch prediction failed.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-md mx-auto p-6 bg-white rounded shadow mt-10 font-sans">
      <h1 className="text-2xl font-bold mb-6 text-center">Batch Prediction</h1>

      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label htmlFor="csvFile" className="block mb-1 font-semibold">
            Upload CSV File
          </label>
          <input
            id="csvFile"
            type="file"
            accept=".csv"
            onChange={handleCsvChange}
            required
          />
        </div>

        <div>
          <label htmlFor="zipFile" className="block mb-1 font-semibold">
            Upload ZIP of Images
          </label>
          <input
            id="zipFile"
            type="file"
            accept=".zip"
            onChange={handleZipChange}
            required
          />
        </div>

        {error && <div className="text-red-600 font-medium">{error}</div>}
        {successMessage && (
          <div className="text-green-600 font-medium">{successMessage}</div>
        )}

        <div className="flex space-x-4">
          <button
            type="submit"
            disabled={loading}
            className="flex-grow bg-rose-600 text-white py-2 rounded hover:bg-rose-700 disabled:opacity-60"
          >
            {loading ? "Processing..." : "Submit Batch"}
          </button>
          <button
            type="button"
            onClick={handleReset}
            disabled={loading}
            className="flex-grow bg-gray-400 text-white py-2 rounded hover:bg-gray-500 disabled:opacity-60"
          >
            Reset
          </button>
        </div>
      </form>
    </div>
  );
};

export default BatchPredictionPage;

import React, { useState } from "react";
import { auth } from "../utils/firebase";
import api from "../utils/api";

const PredictionPage = () => {
  const [text, setText] = useState("");
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<any>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);
  const [showExplanation, setShowExplanation] = useState(false);

  const handleImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setImageFile(e.target.files[0]);
    }
  };

  const handleReset = () => {
    setText("");
    setImageFile(null);
    setError(null);
    setResult(null);
    setSuccessMessage(null);
    setShowExplanation(false);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    setError(null);
    setResult(null);
    setSuccessMessage(null);
    setShowExplanation(false);

    if (!text.trim()) {
      setError("Please enter some text.");
      return;
    }

    if (!imageFile) {
      setError("Please upload an image.");
      return;
    }

    const user = auth.currentUser;
    if (!user) {
      setError("You must be logged in to make a prediction.");
      return;
    }

    setLoading(true);

    try {
      const token = await user.getIdToken();

      const formData = new FormData();
      formData.append("text", text);
      formData.append("image", imageFile);

      const response = await api.post("/predict", formData, {
        headers: {
          "Content-Type": "multipart/form-data",
          Authorization: `Bearer ${token}`,
        },
      });

      setResult(response.data);
      setSuccessMessage("Prediction completed successfully!");
    } catch (err: any) {
      console.error(err);
      setError(err.response?.data?.message || "Prediction failed.");
    } finally {
      setLoading(false);
    }
  };

  // Helper to render explanations for each category
  const renderExplanation = (category: string) => {
    const textExp = result?.text_explanations?.[category];
    const imgExp = result?.image_explanations?.[category];

    return (
      <div key={category} className="mb-6">
        <h3 className="font-semibold text-lg capitalize mb-2">{category} Explanation</h3>

        {textExp ? (
          Array.isArray(textExp) ? (
            <ul className="list-disc list-inside max-h-40 overflow-auto text-sm mb-4">
              {textExp.map(
                (item: { word: string; weight: number }, idx: number) => (
                  <li key={idx}>
                    <span className="font-mono">{item.word}</span>:{" "}
                    {item.weight.toFixed(4)}
                  </li>
                )
              )}
            </ul>
          ) : (
            <p className="whitespace-pre-wrap text-sm mb-4">{textExp}</p>
          )
        ) : (
          <p className="text-sm italic mb-4">No text explanation available.</p>
        )}

        {imgExp ? (
          <img
            src={`data:image/png;base64,${imgExp}`}
            alt={`${category} explanation heatmap`}
            className="max-w-full rounded border"
          />
        ) : (
          <p className="text-sm italic">No image explanation available.</p>
        )}
      </div>
    );
  };

  return (
    <div className="max-w-md mx-auto p-6 bg-white rounded shadow mt-10 font-sans">
      <h1 className="text-2xl font-bold mb-6 text-center">Make a Prediction</h1>

      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label htmlFor="text" className="block mb-1 font-semibold">
            Text Input
          </label>
          <textarea
            id="text"
            value={text}
            onChange={(e) => setText(e.target.value)}
            rows={4}
            className="w-full border rounded p-2 focus:outline-none focus:ring-2 focus:ring-rose-500"
            placeholder="Enter your text here"
            required
          />
        </div>

        <div>
          <label htmlFor="image" className="block mb-1 font-semibold">
            Upload Image
          </label>
          <input
            id="image"
            type="file"
            accept="image/*"
            onChange={handleImageChange}
            required
          />
          {imageFile && (
            <div className="mt-4">
              <p className="font-semibold mb-1">Image Preview:</p>
              <img
                src={URL.createObjectURL(imageFile)}
                alt="Preview"
                className="max-w-full max-h-60 rounded border"
              />
            </div>
          )}
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
            {loading ? "Predicting..." : "Predict"}
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

      {result && (
        <div className="mt-6 p-4 bg-gray-100 rounded space-y-5">
          <h2 className="font-semibold text-lg mb-2">Prediction Result:</h2>

          <div>
            <strong>Informative:</strong>{" "}
            <span
              className={`inline-block px-3 py-1 rounded-full font-semibold ${
                result.informative === "informative"
                  ? "bg-green-200 text-green-800"
                  : "bg-gray-300 text-gray-700"
              }`}
            >
              {result.informative}
            </span>
          </div>

          {result.humanitarian && (
            <div>
              <strong>Humanitarian Category:</strong>{" "}
              <span className="inline-block px-3 py-1 rounded-full bg-blue-200 text-blue-800 font-semibold">
                {result.humanitarian}
              </span>
            </div>
          )}

          {result.damage && (
            <div>
              <strong>Damage Category:</strong>{" "}
              <span className="inline-block px-3 py-1 rounded-full bg-red-200 text-red-800 font-semibold">
                {result.damage}
              </span>
            </div>
          )}

          {!showExplanation ? (
            <button
              onClick={() => setShowExplanation(true)}
              className="mt-4 bg-blue-600 text-white py-2 px-4 rounded hover:bg-blue-700"
            >
              Show Explanation
            </button>
          ) : (
            <>
              <button
                onClick={() => setShowExplanation(false)}
                className="mt-4 bg-gray-600 text-white py-2 px-4 rounded hover:bg-gray-700"
              >
                Hide Explanation
              </button>

              <div className="mt-4 max-h-[400px] overflow-auto">
                {/* Loop over available explanation categories */}
                {["informative", "humanitarian", "damage"].map((cat) =>
                  (result.text_explanations?.[cat] || result.image_explanations?.[cat]) ? (
                    renderExplanation(cat)
                  ) : null
                )}
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
};

export default PredictionPage;

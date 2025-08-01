import React, {ChangeEvent, FormEvent, useState} from "react";
import {useNavigate} from "react-router-dom";
import {auth} from "../utils/firebase";
import api from "../utils/api";
import NavBar from '../components/Navbar';
import ImageKit from "imagekit-javascript";
import {UploadResponse} from "imagekit-javascript/dist/src/interfaces";

type PredictionResult = {
    id: number; informative: string; humanitarian?: string; damage?: string; text_explanations?: {
        [key: string]: string | Array<{ word: string; weight: number }>;
    }; image_explanations?: {
        [key: string]: string;
    };
};

type ExplanationCategory = "informative" | "humanitarian" | "damage";

const imagekit = new ImageKit({
    urlEndpoint: process.env.REACT_APP_IMAGEKIT_URL_ENDPOINT!, publicKey: process.env.REACT_APP_IMAGEKIT_PUBLIC_KEY!,
});

const PredictionPage: React.FC = () => {
    const navigate = useNavigate();
    const [text, setText] = useState("");
    const [imageFile, setImageFile] = useState<File | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [result, setResult] = useState<PredictionResult | null>(null);
    const [successMessage, setSuccessMessage] = useState<string | null>(null);
    const [showExplanation, setShowExplanation] = useState(false);
    const [feedback, setFeedback] = useState({
        correct: "" as "yes" | "no" | "",
        comments: "",
        loading: false,
        error: null as string | null,
        success: null as string | null,
        submitted: false,
    });

    const handleImageChange = (e: ChangeEvent<HTMLInputElement>) => {
        if (e.target.files?.[0]) {
            setImageFile(e.target.files[0]);
        }
    };

    const resetForm = () => {
        setText("");
        setImageFile(null);
        setError(null);
        setResult(null);
        setSuccessMessage(null);
        setShowExplanation(false);
        setFeedback({correct: "", comments: "", loading: false, error: null, success: null, submitted: false});
        const fileInput = document.getElementById('imageUpload') as HTMLInputElement;
        if (fileInput) fileInput.value = "";
    };

    const handlePredictionSubmit = async (e: FormEvent) => {
        e.preventDefault();
        setError(null);
        setResult(null);
        setSuccessMessage(null);
        setShowExplanation(false);

        if (!text.trim()) return setError("Please enter a text description.");
        if (!imageFile) return setError("Please upload an image file.");

        const user = auth.currentUser;
        if (!user) return setError("You must be logged in to make a prediction.");

        setLoading(true);
        try {
            const authResponse = await api.get("/imagekit-auth");
            const {signature, expire, token} = authResponse.data;

            const uploadResult: UploadResponse = await imagekit.upload({
                file: imageFile, fileName: imageFile.name, signature, token, expire, useUniqueFileName: true,
            });

            const userToken = await user.getIdToken();
            const predictionFormData = new FormData();
            predictionFormData.append("text", text);
            predictionFormData.append("image_url", uploadResult.url);

            const response = await api.post("/predict", predictionFormData, {
                headers: {Authorization: `Bearer ${userToken}`},
            });

            setResult(response.data);
            setSuccessMessage("Prediction completed successfully!");
        } catch (err: any) {
            setError(err.response?.data?.detail || err.message || "An unexpected error occurred.");
        } finally {
            setLoading(false);
        }
    };

    const handleFeedbackSubmit = async (e: FormEvent) => {
        e.preventDefault();
        if (!feedback.correct) {
            return setFeedback((prev) => ({...prev, error: "Please select whether the prediction was correct."}));
        }
        const user = auth.currentUser;
        if (!user) {
            return setFeedback((prev) => ({...prev, error: "You must be logged in."}));
        }

        setFeedback((prev) => ({...prev, loading: true, error: null}));
        try {
            const token = await user.getIdToken();
            await api.post("/feedback", {
                prediction_id: Number(result?.id),
                correct: feedback.correct === "yes",
                comments: feedback.comments.trim() || null,
            }, {headers: {Authorization: `Bearer ${token}`}});
            setFeedback((prev) => ({...prev, success: "Thank you for your feedback!", submitted: true}));
        } catch (err: any) {
            setFeedback((prev) => ({...prev, error: err.response?.data?.detail || "Failed to submit feedback."}));
        } finally {
            setFeedback((prev) => ({...prev, loading: false}));
        }
    };

    const renderExplanation = (category: ExplanationCategory) => {
        const textExp = result?.text_explanations?.[category];
        const imgExp = result?.image_explanations?.[category];

        return (<div key={category} className="mb-6 p-4 bg-slate-900/50 rounded-lg border border-slate-700">
            <h4 className="font-semibold text-lg capitalize mb-3 text-cyan-400">{category} Explanation</h4>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                    <h5 className="font-medium text-slate-300 mb-2">Text Analysis</h5>
                    {textExp && Array.isArray(textExp) ? (<div
                        className="text-sm text-slate-400 space-y-1 max-h-48 overflow-auto pr-2 custom-scrollbar">
                        {textExp.map((item, idx) => (<div key={idx}>
                            <span className="font-mono text-slate-200">{item.word}</span>: <span
                            className="text-fuchsia-400">{item.weight.toFixed(4)}</span>
                        </div>))}
                    </div>) : <p className="text-sm italic text-slate-500">No text explanation available.</p>}
                </div>
                <div>
                    <h5 className="font-medium text-slate-300 mb-2">Image Analysis</h5>
                    {imgExp ? <img src={`data:image/png;base64,${imgExp}`} alt={`${category} explanation heatmap`}
                                   className="w-full rounded-md border border-slate-600 shadow-lg"/> :
                        <p className="text-sm italic text-slate-500">No image explanation available.</p>}
                </div>
            </div>
        </div>);
    };

    return (<div className="min-h-screen bg-slate-900 text-white flex flex-col font-sans">
        <NavBar/>
        <main className="w-full max-w-4xl mx-auto px-6 py-12 flex-grow">
            <div className="bg-slate-800/50 p-8 rounded-xl border border-slate-700 shadow-2xl shadow-slate-950/50">
                <header className="text-center mb-8">
                    <h1 className="text-4xl md:text-5xl font-bold bg-gradient-to-r from-fuchsia-500 to-cyan-500 bg-clip-text text-transparent pb-2">
                        Make a Prediction
                    </h1>
                    <p className="text-slate-400 mt-2">Upload an image and text for AI-powered disaster
                        classification.</p>
                </header>

                <form onSubmit={handlePredictionSubmit} className="space-y-6">
                    <div>
                        <label htmlFor="textInput" className="block text-slate-300 font-medium mb-2">Text
                            Description</label>
                        <textarea id="textInput" value={text} onChange={(e) => setText(e.target.value)} rows={4}
                                  className="w-full px-4 py-3 bg-slate-900/50 border border-slate-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-fuchsia-500 focus:border-fuchsia-500 transition"
                                  placeholder="e.g., 'A building has collapsed after a powerful earthquake...'"
                                  required/>
                    </div>
                    <div>
                        <label className="block text-slate-300 font-medium mb-2">Disaster Image</label>
                        <label
                            className="group flex flex-col items-center justify-center w-full h-32 border-2 border-dashed border-slate-600 rounded-lg cursor-pointer bg-slate-900/50 hover:bg-slate-800/50 hover:border-fuchsia-500 transition-colors">
                            <div className="flex flex-col items-center justify-center pt-5 pb-6 text-slate-400">
                                <svg className="w-8 h-8 mb-3 group-hover:text-fuchsia-400 transition-colors"
                                     fill="none" stroke="currentColor" viewBox="0 0 24 24"
                                     xmlns="http://www.w3.org/2000/svg">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2"
                                          d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"></path>
                                </svg>
                                <p className="text-sm"><span
                                    className="font-semibold text-fuchsia-400">Click to upload</span> or
                                    drag and drop</p>
                                <p className="text-xs mt-1">{imageFile ? imageFile.name : 'PNG, JPG, or JPEG'}</p>
                            </div>
                            <input id="imageUpload" type="file" accept="image/png, image/jpeg, image/jpg"
                                   onChange={handleImageChange} className="hidden"/>
                        </label>
                        {imageFile && (<div className="mt-4">
                            <p className="font-medium text-slate-400 mb-2 text-center">Preview:</p>
                            <div className="flex justify-center">
                                <img
                                    src={URL.createObjectURL(imageFile)}
                                    alt="Preview"
                                    className="max-h-60 rounded-lg border border-slate-700 shadow-md"
                                />
                            </div>
                        </div>)}

                    </div>

                    {error && <div
                        className="p-3 bg-red-500/10 text-red-400 border border-red-500/30 rounded-lg font-medium">{error}</div>}
                    {successMessage && <div
                        className="p-3 bg-green-500/10 text-green-400 border border-green-500/30 rounded-lg font-medium">{successMessage}</div>}

                    <div className="flex flex-col sm:flex-row space-y-4 sm:space-y-0 sm:space-x-4 pt-4">
                        <button type="submit" disabled={loading}
                                className="w-full bg-fuchsia-600 text-white py-3 rounded-lg hover:bg-fuchsia-700 disabled:opacity-60 disabled:cursor-not-allowed transition shadow-lg shadow-fuchsia-600/20 hover:shadow-fuchsia-600/30 flex items-center justify-center font-semibold">
                            {loading ? <>
                                <svg className="animate-spin -ml-1 mr-3 h-5 w-5" xmlns="http://www.w3.org/2000/svg"
                                     fill="none" viewBox="0 0 24 24">
                                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor"
                                            strokeWidth="4"></circle>
                                    <path className="opacity-75" fill="currentColor"
                                          d="M4 12a8 8 0 018-8v8z"></path>
                                </svg>
                                Processing...</> : "Predict"}
                        </button>
                        <button type="button" onClick={resetForm} disabled={loading}
                                className="w-full bg-slate-700 text-slate-200 py-3 rounded-lg hover:bg-slate-600 disabled:opacity-60 disabled:cursor-not-allowed transition font-semibold">Reset
                            Form
                        </button>
                    </div>
                </form>

                {result && (<section className="mt-10 pt-8 border-t border-slate-700">
                    <h2 className="text-2xl font-semibold mb-6 text-slate-100 text-center">Prediction
                        Results</h2>
                    <div className="mb-6 p-6 bg-slate-900/50 rounded-lg border border-slate-700 space-y-3">
                        <p><strong>Informative Class:</strong> <span
                            className="font-semibold text-cyan-400">{result.informative}</span></p>
                        {result.humanitarian && <p><strong>Humanitarian Class:</strong> <span
                            className="font-semibold text-cyan-400">{result.humanitarian}</span></p>}
                        {result.damage && <p><strong>Damage Class:</strong> <span
                            className="font-semibold text-cyan-400">{result.damage}</span></p>}
                    </div>
                    <div className="text-center mb-6">
                        <button type="button" onClick={() => setShowExplanation(!showExplanation)}
                                className="text-fuchsia-400 hover:text-fuchsia-300 font-medium focus:outline-none transition">
                            {showExplanation ? "Hide" : "Show"} Explanations
                        </button>
                    </div>
                    {showExplanation && <div
                        className="space-y-4">{renderExplanation("informative")}{result.humanitarian && renderExplanation("humanitarian")}{result.damage && renderExplanation("damage")}</div>}

                    {feedback.submitted ? (<div
                        className="mt-8 p-4 text-center bg-green-500/10 text-green-400 border border-green-500/30 rounded-lg">{feedback.success}</div>) : (
                        <form onSubmit={handleFeedbackSubmit}
                              className="mt-8 pt-8 border-t border-slate-700 space-y-4">
                            <h3 className="text-xl font-semibold text-slate-100 text-center">Was this prediction
                                correct?</h3>
                            <div className="flex justify-center space-x-6">
                                <label className="flex items-center space-x-2 cursor-pointer"><input
                                    type="radio" name="correct" value="yes" checked={feedback.correct === "yes"}
                                    onChange={() => setFeedback((prev) => ({
                                        ...prev, correct: "yes", error: null
                                    }))}
                                    className="form-radio bg-slate-700 border-slate-500 text-fuchsia-500 focus:ring-fuchsia-500"
                                    required/><span>Yes</span></label>
                                <label className="flex items-center space-x-2 cursor-pointer"><input
                                    type="radio" name="correct" value="no" checked={feedback.correct === "no"}
                                    onChange={() => setFeedback((prev) => ({
                                        ...prev, correct: "no", error: null
                                    }))}
                                    className="form-radio bg-slate-700 border-slate-500 text-fuchsia-500 focus:ring-fuchsia-500"/><span>No</span></label>
                            </div>
                            <div>
                                <label htmlFor="comments" className="block text-slate-300 font-medium mb-2">Comments
                                    (optional)</label>
                                <textarea id="comments" rows={3} value={feedback.comments}
                                          onChange={(e) => setFeedback((prev) => ({
                                              ...prev, comments: e.target.value
                                          }))} placeholder="Provide additional details or corrections..."
                                          className="w-full px-4 py-3 bg-slate-900/50 border border-slate-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-fuchsia-500 focus:border-fuchsia-500 transition resize-y"/>
                            </div>
                            {feedback.error &&
                                <div className="text-red-400 font-medium text-center">{feedback.error}</div>}
                            <div className="text-center pt-2">
                                <button type="submit" disabled={feedback.loading}
                                        className="bg-cyan-600 text-white px-6 py-3 rounded-lg hover:bg-cyan-700 transition shadow-md flex items-center justify-center disabled:opacity-60 disabled:cursor-not-allowed mx-auto font-semibold">
                                    {feedback.loading ? <>
                                        <svg className="animate-spin -ml-1 mr-3 h-5 w-5"
                                             xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                                            <circle className="opacity-25" cx="12" cy="12" r="10"
                                                    stroke="currentColor" strokeWidth="4"></circle>
                                            <path className="opacity-75" fill="currentColor"
                                                  d="M4 12a8 8 0 018-8v8z"></path>
                                        </svg>
                                        Sending...</> : "Submit Feedback"}
                                </button>
                            </div>
                        </form>)}
                </section>)}
            </div>
        </main>
    </div>);
};

export default PredictionPage;
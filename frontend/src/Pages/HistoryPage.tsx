import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { auth } from "../utils/firebase";
import api from "../utils/api";
import NavBar from '../components/Navbar';

interface Feedback {
    correct: 'yes' | 'no';
    comments: string | null;
    submitted_at: string;
}

interface HistoryItem {
    id: number;
    text: string;
    image_path: string;
    informative: 'informative' | 'not_informative';
    humanitarian: string | null;
    damage: string | null;
    created_at: string;
    feedback: Feedback | null;
}

const HistoryPage: React.FC = () => {
    const [history, setHistory] = useState<HistoryItem[]>([]);
    const [loading, setLoading] = useState<boolean>(true);
    const [error, setError] = useState<string | null>(null);
    const navigate = useNavigate();

    useEffect(() => {
        const fetchHistory = async () => {
            const unsubscribe = auth.onAuthStateChanged(async (user) => {
                if (user) {
                    try {
                        const token = await user.getIdToken();
                        const response = await api.get('/history', {
                            headers: { Authorization: `Bearer ${token}` },
                        });
                        const sortedHistory = response.data.sort(
                            (a: HistoryItem, b: HistoryItem) =>
                                new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
                        );
                        setHistory(sortedHistory);
                    } catch (err: any) {
                        setError(err.response?.data?.message || "Failed to fetch prediction history.");
                    } finally {
                        setLoading(false);
                    }
                } else {
                    setError("You must be logged in to view your history.");
                    setLoading(false);
                }
            });
            return () => unsubscribe();
        };
        fetchHistory();
    }, []);

    const renderFeedback = (feedback: Feedback | null) => {
        if (!feedback) {
            return (
                <div className="text-sm italic text-slate-500 mt-4 text-center">No feedback submitted.</div>
            );
        }
        const isCorrect = feedback.correct === 'yes';
        const borderColor = isCorrect ? 'border-green-500' : 'border-red-500';
        const textColor = isCorrect ? 'text-green-400' : 'text-red-400';

        return (
            <div className={`mt-4 p-4 rounded-md border-l-4 bg-slate-900/50 ${borderColor}`}>
                <p className="font-semibold text-slate-300">
                    Feedback: <span className={textColor}>{isCorrect ? 'Correct' : 'Incorrect'} Prediction</span>
                </p>
                {feedback.comments && <p className="text-slate-400 mt-2 italic">"{feedback.comments}"</p>}
                <p className="text-xs text-slate-500 mt-2 text-right">{new Date(feedback.submitted_at).toLocaleString()}</p>
            </div>
        );
    };

    const renderTag = (label: string, value: string) => {
        const category = label.toLowerCase();
        let colors = 'bg-slate-700 text-slate-300';
        if (category === 'informative' && value.toLowerCase() === 'informative') colors = 'bg-green-600/20 text-green-300 border border-green-500/30';
        if (category === 'humanitarian') colors = 'bg-blue-600/20 text-blue-300 border border-blue-500/30';
        if (category === 'damage') colors = 'bg-yellow-600/20 text-yellow-300 border border-yellow-500/30';

        return (
            <div className="flex justify-between items-center text-sm">
                <span className="text-slate-400">{label}:</span>
                <span className={`px-2 py-1 rounded-md font-semibold ${colors} capitalize`}>
                    {value.replace(/_/g, ' ')}
                </span>
            </div>
        );
    };

    return (
        <div className="min-h-screen bg-slate-900 text-white flex flex-col font-sans">
            <NavBar />
            <main className="flex-grow w-full max-w-7xl mx-auto px-6 py-12">
                <header className="mb-12 text-center">
                    <h1 className="text-4xl md:text-5xl font-bold bg-gradient-to-r from-fuchsia-500 to-cyan-500 bg-clip-text text-transparent pb-2">
                        Prediction History
                    </h1>
                    <p className="mt-3 text-lg text-slate-400 max-w-2xl mx-auto">Review your past analyses and feedback to track model performance.</p>
                </header>

                {loading ? (
                    <div className="text-center text-slate-400 text-lg flex items-center justify-center pt-20">
                        <svg className="animate-spin -ml-1 mr-3 h-5 w-5" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z"></path></svg>
                        Loading history...
                    </div>
                ) : error ? (
                    <div className="p-4 text-center bg-red-500/10 text-red-400 border border-red-500/30 rounded-lg font-medium max-w-lg mx-auto">{error}</div>
                ) : history.length === 0 ? (
                    <div className="text-center bg-slate-800/50 border border-slate-700 p-12 rounded-xl max-w-xl mx-auto mt-10">
                        <h2 className='text-2xl font-bold text-slate-200 mb-4'>No History Found</h2>
                        <p className='mb-6 text-slate-400'>It looks like you haven't made any predictions yet. Get started by analyzing an image and text.</p>
                        <button onClick={() => navigate('/predict')} className="bg-fuchsia-600 text-white px-6 py-3 rounded-lg text-lg font-semibold shadow-lg shadow-fuchsia-600/30 hover:bg-fuchsia-700 transform hover:-translate-y-1 transition-all">
                            Make a Prediction
                        </button>
                    </div>
                ) : (
                    <div className="grid gap-8 grid-cols-1 md:grid-cols-2 lg:grid-cols-3">
                        {history.map((item, index) => (
                            <article key={item.id} className="bg-slate-800/50 rounded-xl border border-slate-700 flex flex-col overflow-hidden hover:border-fuchsia-500/50 hover:-translate-y-1 transition-all duration-300 shadow-lg hover:shadow-xl hover:shadow-fuchsia-900/20">
                                <div className="relative h-48 w-full bg-slate-700">
                                    <img src={item.image_path} alt={`Prediction ${item.id}`} className="object-cover w-full h-full" onError={(e) => { (e.target as HTMLImageElement).src = "https://placehold.co/400x300/1e293b/94a3b8?text=Image+Not+Found"; }} />
                                    <time className="absolute top-2 right-2 bg-black/50 text-white text-xs px-2 py-1 rounded backdrop-blur-sm" dateTime={item.created_at}>{new Date(item.created_at).toLocaleDateString()}</time>
                                </div>
                                <div className="p-5 flex flex-col flex-grow">
                                    <h2 className="text-lg font-semibold text-slate-100 mb-3">Prediction #{history.length - index}</h2>
                                    <blockquote className="text-slate-300 bg-slate-900/50 p-3 rounded-md mb-4 border-l-4 border-cyan-500 italic text-sm">"{item.text}"</blockquote>
                                    <div className="space-y-2 mb-4 flex-grow">
                                        {renderTag('Informative', item.informative)}
                                        {item.humanitarian && renderTag('Humanitarian', item.humanitarian)}
                                        {item.damage && renderTag('Damage', item.damage)}
                                    </div>
                                    {renderFeedback(item.feedback)}
                                </div>
                            </article>
                        ))}
                    </div>
                )}
            </main>
        </div>
    );
};

export default HistoryPage;
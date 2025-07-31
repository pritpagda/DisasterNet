import React from "react";
import {useNavigate} from "react-router-dom";
import {auth} from "../utils/firebase";
import {useAuthState} from "react-firebase-hooks/auth";
import NavBar from '../components/Navbar';

const features = [{
    icon: "🔥",
    title: "AI-Powered Analysis",
    description: "Leverage cutting-edge models to identify disaster type and damage level using both text and image data.",
}, {
    icon: "🧠",
    title: "Explainable Results",
    description: "Understand predictions with visual and textual explanations that highlight contributing factors.",
}, {
    icon: "💬",
    title: "Feedback Loop",
    description: "Provide feedback to continuously improve the model and ensure more reliable outputs.",
},];

const dashboardLinks = [{
    title: "Predict",
    path: "/predict",
    icon: (
        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor"
             className="w-8 h-8 mb-2">
            <path strokeLinecap="round" strokeLinejoin="round"
                  d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 01-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 013.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 013.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 01-3.09 3.09zM18.259 8.715L18 9.75l-.259-1.035a3.375 3.375 0 00-2.455-2.456L14.25 6l1.036-.259a3.375 3.375 0 002.455-2.456L18 2.25l.259 1.035a3.375 3.375 0 002.456 2.456L21.75 6l-1.035.259a3.375 3.375 0 00-2.456 2.456zM16.898 20.572L16.5 21.75l-.398-1.178a3.375 3.375 0 00-2.455-2.456L12.75 18l1.178-.398a3.375 3.375 0 002.455-2.456L16.5 14.25l.398 1.178a3.375 3.375 0 002.456 2.456L20.25 18l-1.178.398a3.375 3.375 0 00-2.456 2.456z"/>
        </svg>),
}, {
    title: "Batch Predict",
    path: "/batch",
    icon: (
        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor"
             className="w-8 h-8 mb-2">
            <path strokeLinecap="round" strokeLinejoin="round"
                  d="M9 12h3.75M9 15h3.75M9 18h3.75m3 .75H18a2.25 2.25 0 002.25-2.25V6.108c0-1.135-.845-2.098-1.976-2.192a48.424 48.424 0 00-1.123-.08m-5.801 0c-.065.21-.1.433-.1.664 0 .414.336.75.75.75h4.5a.75.75 0 00.75-.75c0-.231-.035-.454-.1-.664M6.75 7.5h4.5v.75h-4.5v-.75z"/>
        </svg>),
}, {
    title: "History",
    path: "/history",
    icon: (
        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor"
             className="w-8 h-8 mb-2">
            <path strokeLinecap="round" strokeLinejoin="round" d="M12 6v6h4.5m4.5 0a9 9 0 11-18 0 9 9 0 0118 0z"/>
        </svg>),
},];

const LandingPage: React.FC = () => {
    const [user, loading] = useAuthState(auth);
    const navigate = useNavigate();

    const handleGetStarted = () => {
        if (user) {
            navigate("/predict");
        } else {
            navigate("/login");
        }
    };

    if (loading) {
        return (<div className="flex items-center justify-center min-h-screen bg-slate-900">
            <p className="text-slate-400 text-lg">Loading...</p>
        </div>);
    }

    return (<div className="min-h-screen bg-slate-900 text-white flex flex-col">
        <NavBar/>

        <main className="flex-grow max-w-7xl mx-auto px-6 flex flex-col items-center text-center">
            <section className="mt-16 md:mt-24 max-w-4xl">
                <h2 className="text-5xl md:text-7xl font-extrabold tracking-tighter bg-gradient-to-r from-fuchsia-500 via-red-500 to-orange-500 bg-clip-text text-transparent pb-4">
                    Smarter Disaster Analysis
                </h2>
                <p className="text-lg md:text-xl text-slate-400 mt-4 mb-10">
                    Upload images and text descriptions to classify disasters with explainable AI.
                    Improve safety and response efforts with accurate, interpretable insights.
                </p>
                <button
                    onClick={handleGetStarted}
                    className="bg-fuchsia-600 text-white px-8 py-4 rounded-lg text-lg font-semibold shadow-lg shadow-fuchsia-600/30 hover:bg-fuchsia-700 hover:shadow-xl hover:shadow-fuchsia-600/30 transform hover:-translate-y-1 transition-all duration-300"
                >
                    Get Started Now
                </button>
            </section>

            {user ? (<section className="mt-24 w-full max-w-4xl">
                <h3 className="text-3xl font-bold text-slate-100 mb-8">Dashboard</h3>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
                    {dashboardLinks.map(({title, path, icon}) => (<button
                        key={title}
                        onClick={() => navigate(path)}
                        className="bg-slate-800/50 p-6 rounded-lg border border-slate-700 text-slate-300 hover:border-cyan-500/50 hover:text-white hover:-translate-y-1 transition-all duration-300 flex flex-col items-center justify-center"
                    >
                        {icon}
                        <span className="text-xl font-semibold">{title}</span>
                    </button>))}
                </div>
            </section>) : (<p className="mt-24 text-slate-500 italic">
                Please login to access the dashboard and prediction features.
            </p>)}

            <section className="mt-24 w-full">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
                    {features.map(({icon, title, description}) => (<div
                        key={title}
                        className="bg-slate-800/50 p-6 rounded-lg border border-slate-700 hover:border-fuchsia-500/50 hover:-translate-y-1 transition-all duration-300"
                    >
                        <div
                            className="flex h-12 w-12 items-center justify-center rounded-full bg-fuchsia-900/50 text-2xl mb-4">{icon}</div>
                        <h4 className="text-xl font-semibold mb-2 text-slate-100">{title}</h4>
                        <p className="text-slate-400">{description}</p>
                    </div>))}
                </div>
            </section>
        </main>
    </div>);
};

export default LandingPage;
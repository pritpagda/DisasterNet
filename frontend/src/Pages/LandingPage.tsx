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
return (
<div className="flex items-center justify-center min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
  <div className="relative">
    <div className="w-16 h-16 border-4 border-fuchsia-500/20 border-t-fuchsia-500 rounded-full animate-spin"></div>
    <div className="absolute inset-0 w-12 h-12 m-2 border-4 border-transparent border-t-cyan-400 rounded-full animate-spin animate-reverse"></div>
  </div>
  <p className="text-slate-400 text-lg ml-4">Loading your experience...</p>
</div>
);
}

return (
<div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 text-white flex flex-col relative overflow-hidden">
  {/* Animated background elements */}
  <div className="absolute inset-0 overflow-hidden pointer-events-none">
    <div className="absolute -top-40 -right-40 w-80 h-80 bg-fuchsia-500/10 rounded-full blur-3xl animate-pulse"></div>
    <div className="absolute top-1/2 -left-40 w-80 h-80 bg-cyan-500/10 rounded-full blur-3xl animate-pulse delay-1000"></div>
    <div className="absolute -bottom-40 left-1/2 w-80 h-80 bg-orange-500/10 rounded-full blur-3xl animate-pulse delay-2000"></div>
  </div>

  <NavBar/>

  <main className="flex-grow max-w-7xl mx-auto px-6 flex flex-col items-center text-center relative z-10">
    {/* Hero Section */}
    <section className="mt-20 md:mt-32 max-w-5xl">
      <div className="mb-6">
        <span className="inline-flex items-center px-4 py-2 rounded-full bg-gradient-to-r from-fuchsia-500/10 to-cyan-500/10 border border-fuchsia-500/20 text-fuchsia-300 text-sm font-medium backdrop-blur-sm">
          ✨ Powered by Advanced AI Technology
        </span>
      </div>

      <h1 className="text-6xl md:text-8xl font-black tracking-tighter mb-6">
        <span className="bg-gradient-to-r from-fuchsia-400 via-pink-400 to-orange-400 bg-clip-text text-transparent">
          Smarter
        </span>
        <br />
        <span className="bg-gradient-to-r from-cyan-400 via-blue-400 to-purple-400 bg-clip-text text-transparent">
          Disaster Analysis
        </span>
      </h1>

      <p className="text-xl md:text-2xl text-slate-300 leading-relaxed mb-12 max-w-3xl mx-auto">
        Transform disaster response with <span className="text-fuchsia-400 font-semibold">AI-powered analysis</span>.
        Upload images and descriptions to get <span className="text-cyan-400 font-semibold">instant and explainable insights</span>
      </p>

      <div className="flex flex-col sm:flex-row gap-4 justify-center items-center">
        <button
          onClick={handleGetStarted}
          className="group relative px-10 py-4 bg-gradient-to-r from-fuchsia-600 to-pink-600 rounded-2xl text-lg font-bold shadow-2xl shadow-fuchsia-500/25 hover:shadow-fuchsia-500/40 transform hover:-translate-y-2 transition-all duration-500 overflow-hidden"
        >
          <div className="absolute inset-0 bg-gradient-to-r from-fuchsia-400 to-pink-400 opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
          <span className="relative flex items-center gap-2">
            Get Started Now
            <svg className="w-5 h-5 group-hover:translate-x-1 transition-transform duration-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
            </svg>
          </span>
        </button>
      </div>
    </section>

    {/* Dashboard Section */}
    {user ? (
      <section className="mt-32 w-full max-w-6xl">
        <div className="text-center mb-16">
          <h2 className="text-4xl md:text-5xl font-bold mb-4">
            <span className="bg-gradient-to-r from-slate-100 to-slate-300 bg-clip-text text-transparent">
              Your Dashboard
            </span>
          </h2>
          <p className="text-slate-400 text-lg">Quick access to all your disaster analysis tools</p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          {dashboardLinks.map(({title, path, icon}, index) => (
            <button
              key={title}
              onClick={() => navigate(path)}
              className="group relative bg-gradient-to-br from-slate-800/80 to-slate-900/80 backdrop-blur-sm p-8 rounded-3xl border border-slate-700/50 hover:border-fuchsia-500/50 transform hover:-translate-y-3 hover:rotate-1 transition-all duration-500 shadow-xl hover:shadow-2xl hover:shadow-fuchsia-500/10"
              style={{ animationDelay: `${index * 150}ms` }}
            >
              <div className="absolute inset-0 bg-gradient-to-br from-fuchsia-500/5 to-cyan-500/5 rounded-3xl opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>

              <div className="relative flex flex-col items-center">
                <div className="p-4 bg-gradient-to-br from-fuchsia-500/20 to-cyan-500/20 rounded-2xl mb-4 group-hover:scale-110 transition-transform duration-300">
                  <div className="text-fuchsia-400 group-hover:text-cyan-400 transition-colors duration-300">
                    {icon}
                  </div>
                </div>
                <h3 className="text-xl font-bold text-slate-100 group-hover:text-white transition-colors duration-300">
                  {title}
                </h3>
                <div className="w-12 h-0.5 bg-gradient-to-r from-fuchsia-500 to-cyan-500 mt-2 scale-x-0 group-hover:scale-x-100 transition-transform duration-300"></div>
              </div>
            </button>
          ))}
        </div>
      </section>
    ) : (
      <div className="mt-32 p-8 bg-gradient-to-r from-slate-800/40 to-slate-900/40 backdrop-blur-sm rounded-3xl border border-slate-700/50 max-w-2xl">
        <div className="text-center">
          <div className="w-16 h-16 bg-gradient-to-br from-fuchsia-500/20 to-cyan-500/20 rounded-full flex items-center justify-center mx-auto mb-4">
            <svg className="w-8 h-8 text-fuchsia-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
            </svg>
          </div>
          <h3 className="text-xl font-semibold text-slate-200 mb-2">Ready to Get Started?</h3>
          <p className="text-slate-400">Please login to access the dashboard and unlock powerful prediction features.</p>
        </div>
      </div>
    )}

    {/* Features Section */}
    <section className="mt-32 w-full max-w-7xl mb-20">
      <div className="text-center mb-16">
        <h2 className="text-4xl md:text-5xl font-bold mb-4">
          <span className="bg-gradient-to-r from-slate-100 to-slate-300 bg-clip-text text-transparent">
            Why Choose Our Platform?
          </span>
        </h2>
        <p className="text-slate-400 text-lg max-w-2xl mx-auto">
          Experience the future of disaster analysis with cutting-edge AI technology designed for accuracy and transparency.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
        {features.map(({icon, title, description}, index) => (
          <div
            key={title}
            className="group relative bg-gradient-to-br from-slate-800/60 to-slate-900/60 backdrop-blur-sm p-8 rounded-3xl border border-slate-700/50 hover:border-fuchsia-500/30 transform hover:-translate-y-2 transition-all duration-700 shadow-xl"
            style={{ animationDelay: `${index * 200}ms` }}
          >
            <div className="absolute inset-0 bg-gradient-to-br from-fuchsia-500/5 to-cyan-500/5 rounded-3xl opacity-0 group-hover:opacity-100 transition-opacity duration-500"></div>

            <div className="relative">
              <div className="flex items-center justify-center w-16 h-16 bg-gradient-to-br from-fuchsia-900/50 to-cyan-900/50 rounded-2xl text-3xl mb-6 group-hover:scale-110 group-hover:rotate-6 transition-all duration-300">
                {icon}
              </div>

              <h3 className="text-2xl font-bold mb-4 text-slate-100 group-hover:text-white transition-colors duration-300">
                {title}
              </h3>

              <p className="text-slate-400 leading-relaxed group-hover:text-slate-300 transition-colors duration-300">
                {description}
              </p>

              <div className="absolute bottom-0 left-0 w-full h-1 bg-gradient-to-r from-fuchsia-500 to-cyan-500 rounded-full scale-x-0 group-hover:scale-x-100 transition-transform duration-500 origin-left"></div>
            </div>
          </div>
        ))}
      </div>
    </section>
  </main>
</div>
);
};

export default LandingPage;
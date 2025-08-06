import React, {useEffect, useState} from "react";
import {useLocation, useNavigate} from "react-router-dom";
import {auth, googleProvider,} from "../utils/firebase";
import {
    browserLocalPersistence,
    browserSessionPersistence,
    createUserWithEmailAndPassword,
    onAuthStateChanged,
    sendEmailVerification,
    setPersistence,
    signInWithEmailAndPassword,
    signInWithPopup,
    signOut,
    User,
} from "firebase/auth";
import {AnimatePresence, motion} from "framer-motion";
import {AlertCircle, CheckCircle, KeyRound, LogIn, Mail, ShieldCheck, UserPlus,} from "lucide-react";

const GoogleIcon = () => (<img
    src="https://upload.wikimedia.org/wikipedia/commons/c/c1/Google_%22G%22_logo.svg"
    alt="Google logo"
    width={20}
    height={20}
    className="inline-block"
/>);

const FormInput = ({
                       icon: Icon, ...props
                   }: {
    icon: React.ElementType;
} & React.InputHTMLAttributes<HTMLInputElement>) => (<div className="relative">
    <Icon className="pointer-events-none absolute left-4 top-1/2 -translate-y-1/2 text-slate-400" size={20}/>
    <input
        {...props}
        className="w-full rounded-lg border border-slate-600 bg-slate-900/50 p-3 pl-12 text-slate-100 transition-colors duration-200 focus:border-fuchsia-500 focus:outline-none focus:ring-2 focus:ring-fuchsia-500/50"
    />
</div>);

const AlertMessage = ({
                          type, message,
                      }: {
    type: "success" | "warning" | "error"; message: string;
}) => {
    const config = {
        success: {Icon: CheckCircle, styles: "bg-green-500/10 text-green-400 border-green-500/30"},
        warning: {Icon: AlertCircle, styles: "bg-yellow-500/10 text-yellow-400 border-yellow-500/30"},
        error: {Icon: AlertCircle, styles: "bg-red-500/10 text-red-400 border-red-500/30"},
    };
    const {Icon, styles} = config[type];
    return (<motion.div
        initial={{opacity: 0, y: -10}}
        animate={{opacity: 1, y: 0}}
        exit={{opacity: 0, y: 10}}
        className={`flex items-center gap-3 rounded-lg p-3 text-sm font-medium border ${styles}`}
    >
        <Icon size={20}/>
        <span>{message}</span>
    </motion.div>);
};

const LoggedInView = ({
                          user, onLogout,
                      }: {
    user: User; onLogout: () => void;
}) => (<div className="flex min-h-screen flex-col items-center justify-center bg-slate-900 p-6 font-sans text-white">
    <motion.div
        initial={{opacity: 0, y: 20}}
        animate={{opacity: 1, y: 0}}
        className="w-full max-w-md rounded-2xl border border-slate-700 bg-slate-800/50 p-8 shadow-2xl shadow-slate-950/50 text-center"
    >
        <ShieldCheck className="mx-auto h-12 w-12 text-cyan-400"/>
        <h1 className="mt-4 text-2xl font-bold text-slate-100">
            Authentication Successful
        </h1>
        <p className="mt-2 text-slate-400">
            Logged in as <br/>
            <span className="font-semibold text-slate-200">{user.email}</span>
        </p>
        <div className="mt-8 space-y-4">
            <button
                onClick={onLogout}
                className="w-full rounded-lg bg-fuchsia-600 px-6 py-3 font-semibold text-white shadow-lg shadow-fuchsia-600/20 transition-all hover:bg-fuchsia-700"
            >
                Logout
            </button>
        </div>
    </motion.div>
</div>);

const AuthForm = () => {
    const [email, setEmail] = useState("");
    const [password, setPassword] = useState("");
    const [result, setResult] = useState<{ type: "success" | "warning" | "error"; message: string } | null>(null);
    const [rememberMe, setRememberMe] = useState(false);
    const [activeTab, setActiveTab] = useState<"signin" | "signup">("signin");
    const [isProcessing, setIsProcessing] = useState(false);
    const navigate = useNavigate();
    const location = useLocation();
    const from = (location.state as any)?.from || "/";

    const handleAuthAction = async (action: "signin" | "signup") => {
        setIsProcessing(true);
        setResult(null);
        try {
            const persistence = rememberMe ? browserLocalPersistence : browserSessionPersistence;
            await setPersistence(auth, persistence);

            if (action === "signup") {
                const userCred = await createUserWithEmailAndPassword(auth, email, password);
                await sendEmailVerification(userCred.user);
                setResult({type: "success", message: "Account created. Please check your inbox to verify your email."});
            } else {
                const userCred = await signInWithEmailAndPassword(auth, email, password);
                if (!userCred.user.emailVerified) {
                    await signOut(auth);
                    setResult({type: "warning", message: "Email not verified. Please check your inbox."});
                } else {
                    setResult({type: "success", message: "Login successful! Redirecting..."});
                    navigate(from, {replace: true});
                }
            }
        } catch (err: any) {
            const errorCode = err.code || "";
            const message = errorCode.replace("auth/", "").replace(/-/g, " ").replace(/\b\w/g, (c: string) => c.toUpperCase());
            setResult({type: "error", message: `Error: ${message || "An unknown error occurred."}`});
        } finally {
            setIsProcessing(false);
        }
    };

    const loginWithGoogle = async () => {
        setIsProcessing(true);
        setResult(null);
        try {
            const persistence = rememberMe ? browserLocalPersistence : browserSessionPersistence;
            await setPersistence(auth, persistence);
            await signInWithPopup(auth, googleProvider);
            setResult({type: "success", message: "Google Login successful! Redirecting..."});
            navigate(from, {replace: true});
        } catch (err: any) {
            setResult({type: "error", message: err.message || "Google login failed."});
        } finally {
            setIsProcessing(false);
        }
    };

    return (<div className="flex min-h-screen items-center justify-center bg-slate-900 p-4 font-sans text-white">
        <div className="w-full max-w-md">
            <motion.div initial={{opacity: 0, y: -20}} animate={{opacity: 1, y: 0}} className="mb-8 text-center">
                <ShieldCheck className="mx-auto h-12 w-12 text-cyan-400"/>
                <h1 className="mt-4 text-3xl font-bold text-slate-100">DisasterNet</h1>
                <p className="text-slate-400 mt-2">Sign in or create an account to continue</p>
            </motion.div>

            <motion.div initial={{opacity: 0, y: 20}} animate={{opacity: 1, y: 0}} transition={{delay: 0.1}}
                        className="rounded-2xl border border-slate-700 bg-slate-800/50 p-8 shadow-2xl shadow-slate-950/50">
                <div className="mb-6 flex rounded-lg bg-slate-900/70 p-1">
                    {["signin", "signup"].map((tab) => (<button
                        key={tab}
                        onClick={() => setActiveTab(tab as any)}
                        className={`relative w-full rounded-md py-2.5 text-sm font-semibold transition-colors ${activeTab === tab ? "text-white" : "text-slate-400 hover:text-slate-200"}`}
                    >
                        <span className="relative z-10">{tab === "signin" ? "Sign In" : "Create Account"}</span>
                        {activeTab === tab && <motion.div layoutId="active-tab"
                                                          className="absolute inset-0 z-0 rounded-md bg-slate-700"/>}
                    </button>))}
                </div>

                <form className="space-y-4" onSubmit={(e) => {
                    e.preventDefault();
                    handleAuthAction(activeTab);
                }}>
                    <FormInput icon={Mail} type="email" placeholder="Email" value={email}
                               onChange={(e) => setEmail(e.target.value)} autoComplete="username" required/>
                    <FormInput icon={KeyRound} type="password" placeholder="Password" value={password}
                               onChange={(e) => setPassword(e.target.value)}
                               autoComplete={activeTab === "signin" ? "current-password" : "new-password"}
                               required/>

                    <div className="flex items-center justify-between text-sm">
                        <label className="flex items-center gap-2 text-slate-400 cursor-pointer">
                            <input type="checkbox" checked={rememberMe}
                                   onChange={(e) => setRememberMe(e.target.checked)}
                                   className="h-4 w-4 rounded border-slate-500 bg-slate-700 text-fuchsia-500 focus:ring-fuchsia-500"/>
                            Remember Me
                        </label>
                        {activeTab === "signin" &&
                            <a href="#" className="font-medium text-cyan-400 hover:underline">Forgot Password?</a>}
                    </div>

                    <AnimatePresence>{result &&
                        <AlertMessage type={result.type} message={result.message}/>}</AnimatePresence>

                    <button type="submit" disabled={!email || !password || isProcessing}
                            className="flex w-full items-center justify-center gap-2 rounded-lg bg-fuchsia-600 px-6 py-3 font-semibold text-white shadow-lg shadow-fuchsia-600/20 transition-all duration-300 hover:bg-fuchsia-700 hover:shadow-xl hover:shadow-fuchsia-600/30 transform hover:-translate-y-0.5 disabled:cursor-not-allowed disabled:bg-slate-600 disabled:shadow-none">
                        {isProcessing ? "Processing..." : activeTab === "signin" ? <><LogIn size={18}/> Sign
                            In</> : <><UserPlus size={18}/> Create Account</>}
                    </button>

                    <div className="relative my-4 flex items-center">
                        <div className="flex-grow border-t border-slate-700"></div>
                        <span className="mx-4 flex-shrink text-sm text-slate-500">OR</span>
                        <div className="flex-grow border-t border-slate-700"></div>
                    </div>

                    <button type="button" onClick={loginWithGoogle} disabled={isProcessing}
                            className="flex w-full items-center justify-center gap-3 rounded-lg border border-slate-600 bg-slate-800 px-6 py-3 font-semibold text-slate-200 transition-colors hover:bg-slate-700 disabled:cursor-not-allowed disabled:opacity-50">
                        <GoogleIcon/> Continue with Google
                    </button>
                </form>
            </motion.div>
        </div>
    </div>);
};

const AuthPage = () => {
    const [user, setUser] = useState<User | null>(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const unsubscribe = onAuthStateChanged(auth, (firebaseUser) => {
            setUser(firebaseUser);
            setLoading(false);
        });
        return () => unsubscribe();
    }, []);

    const handleLogout = async () => {
        await signOut(auth);
    };

    if (loading) {
        return (<div className="flex min-h-screen items-center justify-center bg-slate-900">
            <svg className="animate-spin h-8 w-8 text-fuchsia-500" xmlns="http://www.w3.org/2000/svg" fill="none"
                 viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor"
                        strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z"></path>
            </svg>
        </div>);
    }

    return user && user.emailVerified ? <LoggedInView user={user} onLogout={handleLogout}/> : <AuthForm/>;
};

export default AuthPage;
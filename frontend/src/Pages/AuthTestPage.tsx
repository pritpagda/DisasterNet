import React, { useEffect, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import {
  auth,
  googleProvider,
} from "../utils/firebase";
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
import { AnimatePresence, motion } from "framer-motion";
import {
  AlertCircle,
  CheckCircle,
  Flower2,
  KeyRound,
  LogIn,
  Mail,
  UserPlus,
} from "lucide-react";
import api from "../utils/api";

const GoogleIcon = () => (
  <img
    src="https://upload.wikimedia.org/wikipedia/commons/c/c1/Google_%22G%22_logo.svg"
    alt="Google logo"
    width={20}
    height={20}
    className="inline-block"
  />
);

const FormInput = ({
  icon: Icon,
  ...props
}: {
  icon: React.ElementType;
} & React.InputHTMLAttributes<HTMLInputElement>) => (
  <div className="relative">
    <Icon className="pointer-events-none absolute left-4 top-1/2 -translate-y-1/2 text-stone-400" size={20} />
    <input
      {...props}
      className="w-full rounded-lg border border-stone-300 bg-white p-3 pl-12 text-stone-800 transition-colors duration-200 focus:border-rose-400 focus:outline-none focus:ring-2 focus:ring-rose-200/50"
    />
  </div>
);

const AlertMessage = ({
  type,
  message,
}: {
  type: "success" | "warning" | "error";
  message: string;
}) => {
  const config = {
    success: { Icon: CheckCircle, styles: "bg-green-50 text-green-800" },
    warning: { Icon: AlertCircle, styles: "bg-amber-50 text-amber-800" },
    error: { Icon: AlertCircle, styles: "bg-red-50 text-red-800" },
  };
  const { Icon, styles } = config[type];
  return (
    <motion.div
      initial={{ opacity: 0, y: -10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: 10 }}
      className={`flex items-center gap-3 rounded-lg p-3 text-sm font-medium ${styles}`}
    >
      <Icon size={20} />
      <span>{message}</span>
    </motion.div>
  );
};

const LoggedInView = ({
  user,
  onLogout,
}: {
  user: User;
  onLogout: () => void;
}) => {
  const [secureMessage, setSecureMessage] = useState<string>("");

  const callSecureApi = async () => {
    try {
      const token = await user.getIdToken();
      api.defaults.headers.common["Authorization"] = `Bearer ${token}`;
      const response = await api.get("secure");
      setSecureMessage(JSON.stringify(response.data, null, 2));
    } catch (error: any) {
      console.error("API error:", error);
      setSecureMessage("❌ Failed to fetch secure data");
    }
  };

  return (
    <div className="flex min-h-screen flex-col items-center justify-center bg-stone-100 p-6 font-sans">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="w-full max-w-md rounded-2xl border border-stone-200/80 bg-white p-8 shadow-sm text-center"
      >
        <Flower2 className="mx-auto h-12 w-12 text-rose-500" />
        <h1 className="mt-4 font-serif text-2xl font-bold text-stone-800">
          Welcome Back!
        </h1>
        <p className="mt-2 text-stone-500">
          Logged in as <br />
          <span className="font-semibold text-stone-700">{user.email}</span>
        </p>
        <div className="mt-6 space-y-3">
          <button
            onClick={callSecureApi}
            className="w-full rounded-full bg-green-600 px-6 py-3 font-semibold text-white shadow-lg shadow-green-500/20 transition-all hover:bg-green-700"
          >
            Call Secure API
          </button>
          <button
            onClick={onLogout}
            className="w-full p-3 font-semibold text-stone-500 transition-colors hover:text-stone-800"
          >
            Logout
          </button>
        </div>

        <pre className="mt-6 max-h-48 overflow-auto rounded bg-gray-100 p-4 text-left text-sm text-stone-700">
          {secureMessage || "🔄 Secure response will show here..."}
        </pre>
      </motion.div>
    </div>
  );
};

const AuthForm = () => {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [result, setResult] = useState<
    { type: "success" | "warning" | "error"; message: string } | null
  >(null);
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
      const persistence = rememberMe
        ? browserLocalPersistence
        : browserSessionPersistence;
      await setPersistence(auth, persistence);

      if (action === "signup") {
        const userCred = await createUserWithEmailAndPassword(auth, email, password);
        await sendEmailVerification(userCred.user);
        setResult({
          type: "success",
          message: "Account created. Please check your inbox to verify your email.",
        });
      } else {
        const userCred = await signInWithEmailAndPassword(auth, email, password);
        if (!userCred.user.emailVerified) {
          await signOut(auth);
          setResult({
            type: "warning",
            message: "Email not verified. Please check your inbox.",
          });
        } else {
          setResult({ type: "success", message: "Login successful! Redirecting..." });
          navigate(from, { replace: true });
        }
      }
    } catch (err: any) {
      const errorCode = err.code || "";
      const message = errorCode
        .replace("auth/", "")
        .replace(/-/g, " ")
        .replace(/\b\w/g, (c:  String) => c.toUpperCase());
      setResult({ type: "error", message: `Error: ${message}` });
    } finally {
      setIsProcessing(false);
    }
  };

  const loginWithGoogle = async () => {
    setIsProcessing(true);
    setResult(null);
    try {
      const persistence = rememberMe
        ? browserLocalPersistence
        : browserSessionPersistence;
      await setPersistence(auth, persistence);
      await signInWithPopup(auth, googleProvider);
      setResult({ type: "success", message: "Google Login successful! Redirecting..." });
      navigate(from, { replace: true });
    } catch (err: any) {
      setResult({ type: "error", message: err.message || "Google login failed." });
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="flex min-h-screen items-center justify-center bg-stone-100 p-4 font-sans">
      <div className="w-full max-w-md">
        <motion.div initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }} className="mb-8 text-center">
          <Flower2 className="mx-auto h-12 w-12 text-rose-500" />
          <h1 className="mt-4 font-serif text-3xl font-bold text-stone-800">Firebase Auth Test</h1>
        </motion.div>

        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }} className="rounded-2xl border border-stone-200/80 bg-white p-8 shadow-sm">
          <div className="mb-6 flex rounded-lg bg-stone-100 p-1">
            {["signin", "signup"].map((tab) => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab as any)}
                className={`relative w-full rounded-md py-2.5 text-sm font-semibold transition-colors ${
                  activeTab === tab ? "text-rose-700" : "text-stone-600 hover:text-stone-800"
                }`}
              >
                <span className="relative z-10">{tab === "signin" ? "Sign In" : "Create Account"}</span>
                {activeTab === tab && (
                  <motion.div
                    layoutId="active-tab"
                    className="absolute inset-0 z-0 rounded-md bg-white shadow-sm"
                  />
                )}
              </button>
            ))}
          </div>

          <form
            className="space-y-4"
            onSubmit={(e) => {
              e.preventDefault();
              handleAuthAction(activeTab);
            }}
          >
            <FormInput
              icon={Mail}
              type="email"
              placeholder="Email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              autoComplete="username"
              required
            />
            <FormInput
              icon={KeyRound}
              type="password"
              placeholder="Password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              autoComplete={activeTab === "signin" ? "current-password" : "new-password"}
              required
            />

            <div className="flex items-center justify-between text-sm">
              <label className="flex items-center gap-2 text-stone-600 cursor-pointer">
                <input
                  type="checkbox"
                  checked={rememberMe}
                  onChange={(e) => setRememberMe(e.target.checked)}
                  className="h-4 w-4 rounded border-stone-300 text-rose-600 focus:ring-rose-500"
                />
                Remember Me
              </label>
              {activeTab === "signin" && (
                <a href="#" className="font-medium text-rose-600 hover:underline">
                  Forgot Password?
                </a>
              )}
            </div>

            <AnimatePresence>
              {result && <AlertMessage type={result.type} message={result.message} />}
            </AnimatePresence>

            <button
              type="submit"
              disabled={!email || !password || isProcessing}
              className="flex w-full items-center justify-center gap-2 rounded-full bg-rose-600 px-6 py-3 font-semibold text-white shadow-lg shadow-rose-500/20 transition-all duration-300 hover:bg-rose-700 hover:shadow-xl hover:shadow-rose-500/30 transform hover:-translate-y-0.5 disabled:cursor-not-allowed disabled:bg-stone-400 disabled:shadow-none"
            >
              {isProcessing ? (
                "Processing..."
              ) : activeTab === "signin" ? (
                <>
                  <LogIn size={18} /> Sign In
                </>
              ) : (
                <>
                  <UserPlus size={18} /> Create Account
                </>
              )}
            </button>

            <div className="relative my-4 flex items-center">
              <div className="flex-grow border-t border-stone-200"></div>
              <span className="mx-4 flex-shrink text-sm text-stone-400">OR</span>
              <div className="flex-grow border-t border-stone-200"></div>
            </div>

            <button
              type="button"
              onClick={loginWithGoogle}
              disabled={isProcessing}
              className="flex w-full items-center justify-center gap-3 rounded-full border border-stone-300 bg-white px-6 py-3 font-semibold text-stone-700 transition-colors hover:bg-stone-50 disabled:cursor-not-allowed disabled:bg-stone-100"
            >
              <GoogleIcon /> Continue with Google
            </button>
          </form>
        </motion.div>
      </div>
    </div>
  );
};

const AuthTestPage = () => {
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
    return <div className="flex min-h-screen items-center justify-center bg-stone-100"></div>;
  }

  return user ? <LoggedInView user={user} onLogout={handleLogout} /> : <AuthForm />;
};

export default AuthTestPage;

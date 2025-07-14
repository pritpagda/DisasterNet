import React, { useEffect, useState } from "react";
import { initializeApp } from "firebase/app";
import {
  getAuth,
  onAuthStateChanged,
  signOut,
  signInWithPopup,
  GoogleAuthProvider,
} from "firebase/auth";
import api from "../utils/api";

// ✅ Firebase Config
const firebaseConfig = {
  apiKey: "AIzaSyCIkjGgmB74fHdQq34-QvdqH_jddz80Qhw",
  authDomain: "disaster-feea2.firebaseapp.com",
  projectId: "disaster-feea2",
  storageBucket: "disaster-feea2.firebasestorage.app",
  messagingSenderId: "367609181991",
  appId: "1:367609181991:web:b232d72e71b546deaddb02",
  measurementId: "G-613EPPN70J",
};

// ✅ Firebase Initialization
const firebaseApp = initializeApp(firebaseConfig);
const auth = getAuth(firebaseApp);


// ✅ Attach token to Axios
const setAuthToken = (token: string) => {
  api.defaults.headers.common["Authorization"] = `Bearer ${token}`;
};

const AuthTestPage: React.FC = () => {
  const [user, setUser] = useState<any>(null);
  const [secureMessage, setSecureMessage] = useState<string>("");

  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, async (user) => {
      setUser(user);
      if (user) {
        const token = await user.getIdToken();
        setAuthToken(token);
      }
    });
    return () => unsubscribe();
  }, []);

  const handleGoogleLogin = async () => {
    const provider = new GoogleAuthProvider();
    try {
      await signInWithPopup(auth, provider);
    } catch (err) {
      console.error("Login error:", err);
    }
  };

  const handleLogout = async () => {
    try {
      await signOut(auth);
      setSecureMessage("");
    } catch (err) {
      console.error("Logout error:", err);
    }
  };

  const callSecureApi = async () => {
    try {
      const response = await api.get("secure");
      setSecureMessage(JSON.stringify(response.data, null, 2));
    } catch (error: any) {
      console.error("API error:", error);
      setSecureMessage("❌ Failed to fetch secure data");
    }
  };

  return (
    <div className="p-8 max-w-xl mx-auto text-center">
      <h1 className="text-2xl font-bold mb-6">🔐 Firebase Auth Test</h1>

      {!user ? (
        <button
          onClick={handleGoogleLogin}
          className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700"
        >
          Sign in with Google
        </button>
      ) : (
        <div className="space-y-4">
          <p className="text-green-600 font-semibold">✅ Logged in as: {user.email}</p>
          <div className="flex justify-center gap-4">
            <button
              onClick={callSecureApi}
              className="bg-green-600 text-white px-4 py-2 rounded hover:bg-green-700"
            >
              Call Secure API
            </button>
            <button
              onClick={handleLogout}
              className="bg-gray-500 text-white px-4 py-2 rounded hover:bg-gray-600"
            >
              Logout
            </button>
          </div>
          <pre className="mt-4 bg-gray-100 p-4 rounded text-left overflow-x-auto">
            {secureMessage || "🔄 Secure response will show here..."}
          </pre>
        </div>
      )}
    </div>
  );
};

export default AuthTestPage;

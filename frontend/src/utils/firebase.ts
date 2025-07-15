// utils/firebase.ts

import { initializeApp } from "firebase/app";
import {
  getAuth,
  GoogleAuthProvider,
  setPersistence,
  browserLocalPersistence,
  browserSessionPersistence,
} from "firebase/auth";

// Your Firebase project configuration
const firebaseConfig = {
  apiKey: "AIzaSyCIkjGgmB74fHdQq34-QvdqH_jddz80Qhw",
  authDomain: "disaster-feea2.firebaseapp.com",
  projectId: "disaster-feea2",
  storageBucket: "disaster-feea2.appspot.com", // fixed `.app` → `.com`
  messagingSenderId: "367609181991",
  appId: "1:367609181991:web:b232d72e71b546deaddb02",
  measurementId: "G-613EPPN70J",
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);

// Auth instance
const auth = getAuth(app);

// Google provider instance
const googleProvider = new GoogleAuthProvider();

// Utility function to set persistence mode based on "remember me"
const setAuthPersistence = (rememberMe: boolean) =>
  setPersistence(auth, rememberMe ? browserLocalPersistence : browserSessionPersistence);

export { auth, googleProvider, setAuthPersistence };

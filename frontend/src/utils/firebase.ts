import {initializeApp} from "firebase/app";
import {
    browserLocalPersistence,
    browserSessionPersistence,
    getAuth,
    GoogleAuthProvider,
    setPersistence,
} from "firebase/auth";

const firebaseConfig = {
    apiKey: process.env.REACT_APP_FIREBASE_API_KEY,
    authDomain: process.env.REACT_APP_FIREBASE_AUTH_DOMAIN,
    projectId: process.env.REACT_APP_FIREBASE_PROJECT_ID,
    storageBucket: process.env.REACT_APP_FIREBASE_STORAGE_BUCKET,
    messagingSenderId: process.env.REACT_APP_FIREBASE_MESSAGING_SENDER_ID,
    appId: process.env.REACT_APP_FIREBASE_APP_ID,
    measurementId: process.env.REACT_APP_FIREBASE_MEASUREMENT_ID,
};

const app = initializeApp(firebaseConfig);
const auth = getAuth(app);
const googleProvider = new GoogleAuthProvider();

const setAuthPersistence = (rememberMe: boolean) => setPersistence(auth, rememberMe ? browserLocalPersistence : browserSessionPersistence);

export {auth, googleProvider, setAuthPersistence};
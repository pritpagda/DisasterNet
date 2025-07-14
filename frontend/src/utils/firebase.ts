import {initializeApp} from "firebase/app";
import {getAuth, GoogleAuthProvider} from "firebase/auth";

const firebaseConfig = {
  apiKey: "AIzaSyCIkjGgmB74fHdQq34-QvdqH_jddz80Qhw",
  authDomain: "disaster-feea2.firebaseapp.com",
  projectId: "disaster-feea2",
  storageBucket: "disaster-feea2.firebasestorage.app",
  messagingSenderId: "367609181991",
  appId: "1:367609181991:web:b232d72e71b546deaddb02",
  measurementId: "G-613EPPN70J",
};

const app = initializeApp(firebaseConfig);

export const auth = getAuth(app);
export const googleProvider = new GoogleAuthProvider();
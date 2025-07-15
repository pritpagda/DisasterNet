import React, { useEffect, useState } from "react";
import { Navigate, useLocation } from "react-router-dom";
import { auth } from "../utils/firebase";
import { onAuthStateChanged, type User } from "firebase/auth";

interface ProtectedRouteProps {
  children: React.ReactNode;
}

const ProtectedRoute: React.FC<ProtectedRouteProps> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);
  const location = useLocation();

  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, (firebaseUser) => {
      setUser(firebaseUser);
      setLoading(false);
    });
    return () => unsubscribe();
  }, []);

  if (loading) {
    return <div>Loading...</div>; // Or your spinner/loading UI
  }

  if (!user) {
    // Not authenticated → redirect to login page, preserve requested path for redirect after login
    return <Navigate to="/login" state={{ from: location }} replace />;
  }

  // Authenticated → render children components (the protected page)
  return <>{children}</>;
};

export default ProtectedRoute;

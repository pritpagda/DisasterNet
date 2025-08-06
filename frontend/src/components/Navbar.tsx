import React, {useEffect, useState} from 'react';
import {Link, useLocation, useNavigate} from 'react-router-dom';
import {onAuthStateChanged, signOut, User} from 'firebase/auth';
import {auth} from '../utils/firebase';

const NAV_LINKS = [{name: 'Home', path: '/'}, {name: 'Single Prediction', path: '/predict'}, {
    name: 'Batch Prediction', path: '/batch'
}, {name: 'History', path: '/history'},];

const NavBar: React.FC = () => {
    const navigate = useNavigate();
    const location = useLocation();
    const [user, setUser] = useState<User | null>(null);

    useEffect(() => {
        const unsubscribe = onAuthStateChanged(auth, (currentUser) => {
            setUser(currentUser);
        });
        return () => unsubscribe();
    }, []);

    const handleLogout = async () => {
        try {
            await signOut(auth);
            navigate('/');
        } catch (error) {
            console.error('Error signing out:', error);
        }
    };

    const visibleLinks = NAV_LINKS.filter((link) => link.path !== location.pathname);

    return (<header className="sticky top-0 z-50 w-full border-b border-white/10 bg-slate-900/80 backdrop-blur-md">
        <div className="mx-auto flex h-20 max-w-7xl items-center justify-between px-6">
            <Link
                to="/"
                className="bg-gradient-to-r from-fuchsia-500 to-cyan-500 bg-clip-text text-2xl font-bold text-transparent"
            >
                DisasterNet
            </Link>

            {user ? (<div className="flex items-center gap-4">
                <nav className="hidden items-center space-x-1 md:flex lg:space-x-4">
                    {visibleLinks.map((link) => (<Link
                        key={link.path}
                        to={link.path}
                        className="rounded-md px-3 py-2 text-sm font-medium text-slate-300 transition-colors hover:bg-slate-700 hover:text-white"
                    >
                        {link.name}
                    </Link>))}
                </nav>
                <button
                    onClick={handleLogout}
                    className="rounded-lg bg-slate-800 px-4 py-2 text-sm font-semibold text-white transition-colors hover:bg-red-500/80"
                >
                    Logout
                </button>
            </div>) : (<Link
                to="/login"
                className="rounded-lg bg-fuchsia-600 px-4 py-2 text-sm font-semibold text-white transition-colors hover:bg-fuchsia-700"
            >
                Login
            </Link>)}
        </div>
    </header>);
};

export default NavBar;
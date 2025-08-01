import React, {ChangeEvent, FormEvent, useRef, useState} from 'react';
import {auth} from '../utils/firebase';
import api from '../utils/api';
import NavBar from '../components/Navbar';

interface FileState {
    csv: File | null;
    zip: File | null;
}

interface StatusState {
    loading: boolean;
    error: string | null;
    success: string | null;
}

const BatchPredictionPage: React.FC = () => {
    const [files, setFiles] = useState<FileState>({csv: null, zip: null});
    const [status, setStatus] = useState<StatusState>({
        loading: false, error: null, success: null,
    });

    const csvRef = useRef<HTMLInputElement>(null);
    const zipRef = useRef<HTMLInputElement>(null);

    const handleFileChange = (e: ChangeEvent<HTMLInputElement>, fileType: 'csv' | 'zip') => {
        if (e.target.files?.[0]) {
            setFiles((prev) => ({...prev, [fileType]: e.target.files![0]}));
        }
    };

    const triggerFileInput = (ref: React.RefObject<HTMLInputElement | null>) => {
        ref.current?.click();
    };

    const resetForm = () => {
        setFiles({csv: null, zip: null});
        setStatus({loading: false, error: null, success: null});
        if (csvRef.current) csvRef.current.value = '';
        if (zipRef.current) zipRef.current.value = '';
    };

    const handleSubmit = async (e: FormEvent<HTMLFormElement>) => {
        e.preventDefault();
        setStatus({loading: true, error: null, success: null});

        if (!files.csv || !files.zip) {
            return setStatus({
                loading: false, error: 'Please upload both CSV and ZIP files.', success: null
            });
        }

        const user = auth.currentUser;
        if (!user) {
            return setStatus({
                loading: false, error: 'You must be logged in to submit a batch prediction.', success: null
            });
        }

        try {
            const token = await user.getIdToken();
            const formData = new FormData();
            formData.append('images_zip', files.zip);
            formData.append('texts_csv', files.csv);

            const response = await api.post('/predict-batch', formData, {
                headers: {'Content-Type': 'multipart/form-data', Authorization: `Bearer ${token}`},
                responseType: 'blob',
            });

            const blob = new Blob([response.data], {type: 'text/csv'});
            const url = window.URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.href = url;
            link.setAttribute('download', `batch_predictions_${new Date().toISOString()}.csv`);
            document.body.appendChild(link);
            link.click();
            link.remove();
            window.URL.revokeObjectURL(url);

            setStatus({
                loading: false,
                error: null,
                success: 'Batch prediction completed! Your CSV file has been downloaded.',
            });
        } catch (err: any) {
            const errorMsg = err.response?.data?.message || 'Batch prediction failed. Please check your files and try again.';
            setStatus({loading: false, error: errorMsg, success: null});
        }
    };

    const UploadBox = ({type}: { type: 'csv' | 'zip' }) => {
        const isSelected = !!files[type];
        const isCsv = type === 'csv';

        const baseClasses = "border-2 border-dashed rounded-xl p-6 text-center cursor-pointer transition-all duration-300 flex flex-col items-center justify-center h-full transform hover:-translate-y-1";
        const dynamicClasses = isSelected ? (isCsv ? 'border-solid border-green-500 bg-green-900/40' : 'border-solid border-cyan-500 bg-cyan-900/40') : 'border-slate-600 hover:bg-slate-800/50 hover:border-fuchsia-500';

        const icon = isCsv ? '📄' : '📦';
        const title = isCsv ? 'Upload CSV Text Data' : 'Upload ZIP Image Archive';
        const description = isCsv ? 'Contains image names and text.' : 'Contains all corresponding images.';

        return (<div onClick={() => !status.loading && triggerFileInput(isCsv ? csvRef : zipRef)}
                     className={`${baseClasses} ${dynamicClasses}`}>
            <input ref={isCsv ? csvRef : zipRef} type="file" accept={isCsv ? '.csv' : '.zip'}
                   onChange={(e) => handleFileChange(e, type)} className="hidden"/>
            <div className="text-4xl mb-3">{icon}</div>
            <h3 className={`font-semibold text-lg ${isSelected ? (isCsv ? 'text-green-300' : 'text-cyan-300') : 'text-slate-200'}`}>
                {files[type]?.name || title}
            </h3>
            <p className={`mt-1 text-sm ${isSelected ? (isCsv ? 'text-green-400/80' : 'text-cyan-400/80') : 'text-slate-400'}`}>
                {isSelected ? `(${(files[type]!.size / 1024 / 1024).toFixed(2)} MB) - Click to change` : description}
            </p>
        </div>);
    };

    return (<div className="min-h-screen bg-slate-900 text-white flex flex-col font-sans">
        <NavBar/>
        <main className="w-full max-w-4xl mx-auto px-6 py-12 flex-grow">
            <div className="bg-slate-800/50 p-8 rounded-xl border border-slate-700 shadow-2xl shadow-slate-950/50">
                <header className="text-center mb-8">
                    <h1 className="text-4xl md:text-5xl font-bold bg-gradient-to-r from-fuchsia-500 to-cyan-500 bg-clip-text text-transparent pb-2">
                        Batch Prediction
                    </h1>
                    <p className="text-slate-400 mt-2">Upload your data in bulk for efficient processing.</p>
                </header>

                <section className="mb-8 bg-slate-900/50 border border-slate-700 rounded-lg p-6 text-slate-300">
                    <h2 className="text-lg font-semibold mb-3 text-cyan-400">Instructions</h2>
                    <ul className="list-disc list-inside space-y-2 text-sm">
                        <li>Upload a <b className="text-slate-100">CSV file</b> with two columns: `image path` and
                            `text`.
                        </li>
                        <li>Upload a <b className="text-slate-100">ZIP archive</b> containing all corresponding
                            images.
                        </li>
                        <li>The resulting predictions will be downloaded as a new CSV file automatically upon
                            completion.
                        </li>
                    </ul>
                </section>

                <form onSubmit={handleSubmit} className="space-y-6">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        <UploadBox type="csv"/>
                        <UploadBox type="zip"/>
                    </div>

                    <div className="min-h-[3.5rem] pt-2 flex items-center justify-center">
                        {status.loading && (<div className="flex flex-col items-center justify-center space-y-2">
                            <svg className="animate-spin h-8 w-8 text-fuchsia-500"
                                 xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor"
                                        strokeWidth="4"></circle>
                                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z"></path>
                            </svg>
                            <span className="text-fuchsia-400 font-medium">Processing... Please wait.</span>
                        </div>)}
                        {status.error && <div
                            className="p-3 bg-red-500/10 text-red-400 border border-red-500/30 rounded-lg font-medium">{status.error}</div>}
                        {status.success && <div
                            className="p-3 bg-green-500/10 text-green-400 border border-green-500/30 rounded-lg font-medium">{status.success}</div>}
                    </div>

                    <div
                        className="flex flex-col sm:flex-row space-y-4 sm:space-y-0 sm:space-x-4 pt-6 border-t border-slate-700">
                        <button type="submit" disabled={status.loading || !files.csv || !files.zip}
                                className="w-full bg-fuchsia-600 text-white py-3 rounded-lg hover:bg-fuchsia-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-lg shadow-fuchsia-600/20 font-semibold flex items-center justify-center">
                            {status.loading ? (<>
                                <svg className="animate-spin -ml-1 mr-3 h-5 w-5"
                                     xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor"
                                            strokeWidth="4"></circle>
                                    <path className="opacity-75" fill="currentColor"
                                          d="M4 12a8 8 0 018-8v8z"></path>
                                </svg>
                                Processing...</>) : 'Submit Batch'}
                        </button>
                        <button type="button" onClick={resetForm} disabled={status.loading}
                                className="w-full bg-slate-700 text-slate-200 py-3 rounded-lg hover:bg-slate-600 disabled:opacity-50 disabled:cursor-not-allowed transition-all font-semibold">
                            Reset
                        </button>
                    </div>
                </form>
            </div>
        </main>

    </div>);
};

export default BatchPredictionPage;
"use client";
import React, { useState, useEffect } from 'react';
import NetworkGraph from './NetworkGraph';
import { apiUrl } from '../lib/api';

export default function Dashboard({ entity, setSelectedEntity }: any) {
    const [details, setDetails] = useState<any>(null);
    const [people, setPeople] = useState([]);
    const [events, setEvents] = useState([]);
    const [showNetwork, setShowNetwork] = useState(false);
    const uid = entity?.uid;

    useEffect(() => {
        if (!uid) return;
        const fetchData = async () => {
            try {
                const [detRes, pplRes, evRes] = await Promise.all([
                    fetch(apiUrl(`/api/entity/${uid}`)),
                    fetch(apiUrl(`/api/entity/${uid}/people`)),
                    fetch(apiUrl(`/api/entity/${uid}/events`))
                ]);

                if (detRes.ok) setDetails((await detRes.json()).data);
                if (pplRes.ok) setPeople((await pplRes.json()).people);
                if (evRes.ok) setEvents((await evRes.json()).events);
            } catch (err) {
                console.error("Error fetching entity data", err);
            }
        };
        fetchData();
    }, [uid]);

    if (!uid) {
        return (
            <div className="flex-1 p-8 flex items-center justify-center bg-neutral-950">
                <div className="text-center max-w-lg bg-neutral-900 border border-neutral-800 p-10 rounded-2xl shadow-xl">
                    <span className="text-5xl mb-4 block">🌍</span>
                    <h2 className="text-2xl font-bold text-neutral-100 mb-2">Global View</h2>
                    <p className="text-neutral-400 font-medium leading-relaxed">Use the sidebar to search for and select an entity to investigate, or use the global agent chat to query the entire database.</p>
                </div>
            </div>
        );
    }

    if (!details) {
        return <div className="flex-1 p-10 flex text-purple-400 justify-center font-medium bg-neutral-950 text-lg animate-pulse">Gathering intelligence...</div>;
    }

    return (
        <div className="flex-1 p-10 overflow-y-auto bg-neutral-950 text-neutral-200 custom-scrollbar">
            <div className="flex justify-between items-center mb-10 border-b border-neutral-800 pb-6">
                <h1 className="text-4xl font-black text-white flex items-center gap-4">
                    <span className="bg-purple-900/30 text-purple-500 p-3 rounded-2xl border border-purple-500/30 shadow-lg shadow-purple-900/20">🕵️</span>{details.name || entity.name}
                </h1>
                <button
                    onClick={() => setShowNetwork(!showNetwork)}
                    className="bg-purple-600/20 hover:bg-purple-600 text-purple-300 hover:text-white border border-purple-500/30 font-bold py-3 px-6 rounded-xl transition-all shadow-lg hover:shadow-purple-500/50 flex items-center gap-2"
                >
                    {showNetwork ? 'Hide Network' : 'Show Network 🕸️'}
                </button>
            </div>

            {showNetwork && (
                <div className="mb-10 animate-fade-in">
                    <NetworkGraph uid={uid} setSelectedEntity={setSelectedEntity} />
                </div>
            )}

            <div className="grid grid-cols-2 gap-6 mb-8">
                <MetricCard title="Type" value={entity.real_label || '-'} />
                <MetricCard title="City" value={details.City || '-'} />
            </div>

            <div className="bg-neutral-900 rounded-2xl shadow-xl border border-neutral-800 p-6 mb-10">
                <h3 className="text-sm font-bold text-neutral-500 uppercase tracking-wider mb-4 border-b border-neutral-800 pb-2">At a Glance</h3>
                <div className="grid grid-cols-2 gap-y-3 gap-x-12 text-sm">
                    <MetaRow label="Community" value={details.Community} />
                    <MetaRow label="Legal Form" value={details.legal_form} />
                    <MetaRow label="Deletion Date" value={details.deletion_date} />
                    <MetaRow label="Capital Nominal" value={details.capital_nominal} />
                    <MetaRow label="Capital Paid" value={details.capital_paid} />
                </div>
            </div>

            <div className="bg-neutral-900 rounded-2xl shadow-xl border border-neutral-800 p-8 mb-10">
                <h3 className="text-xl font-bold mb-6 text-white border-b border-neutral-800 pb-3">🏢 Company Profile</h3>
                <p className="mb-4 text-neutral-300 leading-relaxed"><strong className="text-purple-400 font-bold">📍 Address:</strong> {details.address || '-'}</p>
                <p className="text-neutral-300 leading-relaxed"><strong className="text-purple-400 font-bold">🎯 Purpose:</strong> {details.purpose || '-'}</p>
            </div>

            {/* Associated People */}
            <div className="bg-neutral-900 rounded-2xl shadow-xl border border-neutral-800 p-8 mb-10">
                <h3 className="text-xl font-bold mb-6 text-white border-b border-neutral-800 pb-3">👥 Associated People (Management & Board)</h3>
                <div className="overflow-x-auto rounded-xl ring-1 ring-neutral-800 max-h-96 overflow-y-auto custom-scrollbar">
                    <table className="w-full text-left border-collapse">
                        <thead className="sticky top-0 bg-neutral-900 z-10">
                            <tr className="bg-neutral-800/80 border-b border-neutral-700 text-purple-300">
                                <th className="p-4 font-bold uppercase text-xs tracking-wider">Name</th>
                                <th className="p-4 font-bold uppercase text-xs tracking-wider">Origin</th>
                                <th className="p-4 font-bold uppercase text-xs tracking-wider">Since</th>
                                <th className="p-4 font-bold uppercase text-xs tracking-wider">Type</th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-neutral-800">
                            {people.map((p: any, i: number) => (
                                <tr key={i} className="hover:bg-neutral-800/50 transition-colors">
                                    <td className="p-4 text-sm font-bold text-white">{p.Name}</td>
                                    <td className="p-4 text-sm text-neutral-400">{p.Origin || '-'}</td>
                                    <td className="p-4 text-sm text-neutral-400">{p.Since || '-'}</td>
                                    <td className="p-4 text-sm text-neutral-400"><span className="px-2 border border-neutral-700 rounded text-xs bg-neutral-950 text-neutral-300 font-medium">{p.Type || '-'}</span></td>
                                </tr>
                            ))}
                            {people.length === 0 && (
                                <tr><td colSpan={4} className="p-6 text-center text-neutral-500 font-medium">No associated people found via events.</td></tr>
                            )}
                        </tbody>
                    </table>
                </div>
            </div>

            {/* Events */}
            <div className="bg-neutral-900 rounded-2xl shadow-xl border border-neutral-800 p-8 mb-6">
                <h3 className="text-xl font-bold mb-6 text-white border-b border-neutral-800 pb-3">📅 Event History</h3>
                <div className="space-y-4 max-h-96 overflow-y-auto custom-scrollbar pr-2">
                    {events.map((e: any, i: number) => (
                        <div key={i} className="border border-neutral-800 rounded-xl p-5 hover:border-purple-500/50 transition-colors bg-neutral-950">
                            <div className="flex justify-between items-center mb-3">
                                <span className="font-bold text-white flex items-center gap-2"><span className="text-purple-500">❖</span> {e.Rubric}</span>
                                <span className="text-sm bg-neutral-800 text-purple-300 px-3 py-1 rounded-full font-medium border border-neutral-700">{e.Date}</span>
                            </div>
                            {e.Text && <p className="text-sm text-neutral-400 mt-2 bg-neutral-900 p-4 rounded-lg border border-neutral-800 italic leading-relaxed">{e.Text}</p>}
                        </div>
                    ))}
                    {events.length === 0 && (
                        <div className="p-6 text-center text-neutral-500 font-medium border border-neutral-800 rounded-xl bg-neutral-950">No events recorded.</div>
                    )}
                </div>
            </div>
        </div>
    );
}

function MetricCard({ title, value, highlight }: any) {
    return (
        <div className={`p-6 rounded-2xl shadow-sm border flex flex-col items-start transition-all transform hover:-translate-y-1 hover:shadow-lg ${highlight ? 'bg-gradient-to-br from-purple-900/20 to-neutral-900 border-purple-500/50' : 'bg-neutral-900 border-neutral-800'}`}>
            <div className={`text-xs font-bold mb-2 uppercase tracking-wider ${highlight ? 'text-purple-400' : 'text-neutral-500'}`}>{title}</div>
            <div className={`text-xl font-black ${highlight ? 'text-white' : 'text-neutral-200'} break-words w-full`}>{value}</div>
        </div>
    );
}

function MetaRow({ label, value }: { label: string, value: any }) {
    const isEmpty = !value || value === '-';
    return (
        <div className="flex justify-between items-center py-2 border-b border-neutral-800/50 last:border-0">
            <span className="text-neutral-500 font-medium">{label}</span>
            <span className={`font-semibold text-right ${isEmpty ? 'text-neutral-700 italic' : 'text-neutral-200'}`}>
                {isEmpty ? '-' : value}
            </span>
        </div>
    );
}

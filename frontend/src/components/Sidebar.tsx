"use client";
import React, { useState } from 'react';
import { apiUrl } from '../lib/api';
import type { Entity } from '../lib/types';

interface SidebarProps {
    onSelectEntity: (entity: Entity) => void;
    onGlobalSearch: () => void;
}

export default function Sidebar({ onSelectEntity, onGlobalSearch }: SidebarProps) {
    const [searchTerm, setSearchTerm] = useState('');
    const [results, setResults] = useState<Entity[]>([]);
    const [isSearching, setIsSearching] = useState(false);
    const [hasSearched, setHasSearched] = useState(false);

    const handleSearch = async () => {
        if (!searchTerm) return;
        setIsSearching(true);
        setHasSearched(false);
        try {
            const res = await fetch(apiUrl(`/api/search?term=${encodeURIComponent(searchTerm)}`));
            const data = await res.json();
            setResults(data.results || []);
        } catch (error) {
            console.error("Search failed:", error);
            setResults([]);
        } finally {
            setIsSearching(false);
            setHasSearched(true);
        }
    };

    return (
        <div className="w-full h-full bg-neutral-950 text-neutral-100 p-6 flex flex-col gap-6 z-10 overflow-y-auto custom-scrollbar">
            <div className="shrink-0 flex flex-col gap-4">
                <h2 className="text-xl font-bold mb-2 flex items-center gap-2 text-white">
                    <span className="text-2xl text-purple-500">❖</span> SHAB Explorer
                </h2>

                <button
                    onClick={onGlobalSearch}
                    className="w-full bg-purple-600/20 hover:bg-purple-600 text-purple-300 hover:text-white border border-purple-500/30 transition-all font-semibold py-2.5 px-4 rounded-lg flex justify-center items-center gap-2"
                >
                    <span>✦</span> Global View
                </button>

                <div className="relative mt-2">
                    <input
                        type="text"
                        value={searchTerm}
                        onChange={(e) => { setSearchTerm(e.target.value); setHasSearched(false); }}
                        onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
                        placeholder="Type a name to search..."
                        className="w-full pl-4 pr-10 py-3 bg-neutral-900 border border-neutral-800 rounded-lg text-white placeholder-neutral-500 focus:outline-none focus:border-purple-500 focus:ring-1 focus:ring-purple-500 transition-all font-medium text-sm shadow-inner"
                    />
                    <button
                        onClick={handleSearch}
                        className="absolute right-3 top-1/2 transform -translate-y-1/2 text-purple-500 hover:text-purple-400 transition-colors"
                    >
                        {isSearching ? (
                            <div className="w-4 h-4 border-2 border-purple-500 border-t-transparent rounded-full animate-spin"></div>
                        ) : (
                            "➤"
                        )}
                    </button>
                </div>

            </div>

            <div className="flex-1 flex flex-col gap-3 pt-4 border-t border-neutral-800/50">
                {isSearching ? (
                    <div className="text-purple-400 text-sm italic flex justify-center items-center gap-2 mt-4 animate-pulse">
                        <div className="w-4 h-4 border-2 border-purple-500 border-t-transparent rounded-full animate-spin"></div>
                        Searching...
                    </div>
                ) : results.length > 0 ? (
                    results.map((r, i) => (
                        <button
                            key={i}
                            onClick={() => onSelectEntity(r)}
                            className="w-full text-left p-4 rounded-xl bg-neutral-900 border border-neutral-800 hover:border-purple-500 hover:bg-neutral-800 transition-all group"
                        >
                            <div className="font-bold text-sm text-neutral-100 truncate group-hover:text-white drop-shadow-sm">{r.name}</div>
                            <div className="flex justify-between items-center mt-3">
                                <div className="text-[10px] font-bold tracking-wider uppercase text-purple-400 bg-purple-900/40 px-2 py-0.5 rounded border border-purple-500/20">
                                    {r.real_label === 'Company' ? '🏢 ' : r.real_label === 'Person' ? '👤 ' : '🏷️ '} {r.real_label}
                                </div>
                                <div className="text-xs text-neutral-500 truncate max-w-[50%] text-right font-medium" title={r.city || 'Unknown'}>
                                    📍 {r.city || 'Unknown'}
                                </div>
                            </div>
                        </button>
                    ))
                ) : hasSearched ? (
                    <div className="p-4 bg-neutral-900/60 border border-neutral-800 rounded-xl flex flex-col gap-2 mt-4">
                        <span className="text-neutral-400 font-semibold text-sm">No results found for <span className="text-purple-400">&quot;{searchTerm}&quot;</span></span>
                        <span className="text-neutral-500 text-xs leading-relaxed">
                            This search only looks at entity names. Try the <span className="text-purple-400 font-semibold">❖ Global Agent</span> on the right — it can search across all text in the database and may find hidden mentions.
                        </span>
                    </div>
                ) : (
                    <div className="text-neutral-500 text-sm italic text-center mt-4">Type a name and press Enter to search.</div>
                )}
            </div>
        </div>
    );
}

"use client";
import React, { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { apiUrl } from '../lib/api';
import type { ChatMessage, Entity } from '../lib/types';

interface AgentChatProps {
	entity: Entity | null;
	isExpanded: boolean;
	onToggleExpand: () => void;
	onClose: () => void;
}

interface ChatResponse {
	result?: {
		answer?: string;
		cypher?: string;
		data?: Record<string, unknown>[];
	};
}

const EMPTY_MESSAGES: ChatMessage[] = [];

export default function AgentChat({ entity, isExpanded, onToggleExpand, onClose }: AgentChatProps) {
	const [chatHistories, setChatHistories] = useState<Record<string, ChatMessage[]>>({ global: [] });
	const [input, setInput] = useState('');
	const [loading, setLoading] = useState(false);
	const [liveTraceId, setLiveTraceId] = useState<string | null>(null);
	const [liveTrace, setLiveTrace] = useState<string[]>([]);
	const [showLiveTrace, setShowLiveTrace] = useState(false);
	const endOfMessagesRef = useRef<HTMLDivElement>(null);
	const abortControllerRef = useRef<AbortController | null>(null);
	const uid = entity?.uid;
	const name = entity?.name || 'Global';
	const currentSession = uid || 'global';
	const messages = chatHistories[currentSession] ?? EMPTY_MESSAGES;

	useEffect(() => {
		endOfMessagesRef.current?.scrollIntoView({ behavior: "smooth" });
	}, [messages, liveTrace]);

	useEffect(() => {
		let interval: NodeJS.Timeout;
		if (loading && liveTraceId) {
			interval = setInterval(async () => {
				try {
					const res = await fetch(apiUrl(`/api/chat/trace/${liveTraceId}`));
					const data = await res.json();
					if (data.trace) setLiveTrace(data.trace);
				} catch (e) {
					console.error("Error polling trace:", e);
				}
			}, 1000);
		}
		return () => clearInterval(interval);
	}, [loading, liveTraceId]);

	const handleSend = async (overrideInput?: string | React.MouseEvent) => {
		const text = typeof overrideInput === 'string' ? overrideInput : input;
		if (!text.trim() || loading) return;

		const userMsg: ChatMessage = { role: 'user', content: text };
		setChatHistories(prev => ({
			...prev,
			[currentSession]: [...(prev[currentSession] || []), userMsg]
		}));

		if (typeof overrideInput !== 'string') setInput('');
		setLoading(true);
		const newTraceId = Date.now().toString();
		setLiveTraceId(newTraceId);
		setLiveTrace([]);
		setShowLiveTrace(false);

		abortControllerRef.current = new AbortController();

		try {
			const res = await fetch(apiUrl(`/api/chat`), {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ question: userMsg.content, uid, chat_history: messages, trace_id: newTraceId }),
				signal: abortControllerRef.current.signal
			});
			if (!res.ok) throw new Error(`Agent request failed with status ${res.status}`);
			const data = await res.json() as ChatResponse;

			if (data.result?.answer) {
				const assistantMessage: ChatMessage = {
					role: 'assistant',
					content: data.result.answer,
					trace: data.result.cypher,
					records: data.result.data,
				};
				setChatHistories(prev => ({
					...prev,
					[currentSession]: [...(prev[currentSession] || []), assistantMessage]
				}));
			} else {
				setChatHistories(prev => ({
					...prev,
					[currentSession]: [...(prev[currentSession] || []), { role: 'assistant', content: "No answer received." }]
				}));
			}
		} catch (err: unknown) {
			if (err instanceof DOMException && err.name === 'AbortError') {
				setChatHistories(prev => ({
					...prev,
					[currentSession]: [...(prev[currentSession] || []), { role: 'assistant', content: "⚠️ Investigation stopped by user." }]
				}));
			} else {
				setChatHistories(prev => ({
					...prev,
					[currentSession]: [...(prev[currentSession] || []), { role: 'assistant', content: "⚠️ Error investigating." }]
				}));
			}
		} finally {
			setLoading(false);
			setLiveTraceId(null);
			setShowLiveTrace(false);
			abortControllerRef.current = null;
		}
	};

	const handleStop = () => {
		if (abortControllerRef.current) {
			abortControllerRef.current.abort();
		}
	};

	return (
		<div className="w-full h-full flex flex-col bg-neutral-950 relative z-10 shrink-0 overflow-hidden">
			<div className="p-4 border-b border-neutral-800 bg-neutral-900/50 backdrop-blur shrink-0 flex justify-between items-center">
				<h3 className="font-bold text-lg text-white flex items-center gap-2 overflow-hidden">
					<span className="text-purple-500 shrink-0">❖</span>
					{uid ? <span className="truncate" title={`Investigating: ${name}`}>Agent: {name}</span> : 'Global Agent'}
				</h3>
				<div className="flex gap-1 shrink-0 ml-2">
					{onToggleExpand && (
						<button
							onClick={onToggleExpand}
							className="text-neutral-500 hover:text-purple-400 p-1.5 rounded-lg hover:bg-neutral-800 transition-colors flex items-center justify-center font-bold"
							title={isExpanded ? "Shrink Chat" : "Expand Chat"}
						>
							{isExpanded ? '◨' : '◧'}
						</button>
					)}
					{onClose && (
						<button
							onClick={onClose}
							className="text-neutral-500 hover:text-red-400 p-1.5 rounded-lg hover:bg-neutral-800 transition-colors flex items-center justify-center font-bold"
							title="Close Chat"
						>
							✕
						</button>
					)}
				</div>
			</div>

			<div className="flex-1 overflow-y-auto p-4 space-y-4 bg-neutral-950 custom-scrollbar">
				{messages.length === 0 && (
					<div className="text-center mt-10 p-6 bg-neutral-900/50 rounded-xl border border-neutral-800">
						<span className="text-4xl mb-4 block text-neutral-600">⚡️</span>
						<p className="font-medium text-neutral-400 text-sm">
							{uid ? `Ask a question about ${name}...` : "Ex: Find crypto companies in Dubendorf..."}
						</p>
					</div>
				)}
				{messages.map((m, i) => {
					const isLastAssistantMessage = m.role === 'assistant' && i === messages.findLastIndex((msg) => msg.role === 'assistant');

					return (
						<div key={i} className={`flex flex-col w-full ${m.role === 'user' ? 'items-end' : 'items-start'}`}>
							<div className={`max-w-[90%] p-3 rounded-xl text-sm leading-relaxed overflow-x-auto custom-scrollbar ${m.role === 'user'
								? 'user-chat-bubble bg-purple-600 text-white rounded-br-sm'
								: 'bg-neutral-900 text-neutral-200 border border-neutral-800 rounded-bl-sm'
								}`}>
								{m.role === 'user' ? (
									m.content
								) : (
									<ReactMarkdown
										remarkPlugins={[remarkGfm]}
										components={{
										table: ({ children }) => <table className="w-full text-left border-collapse my-2 mt-4 text-xs">{children}</table>,
										thead: ({ children }) => <thead className="bg-neutral-800 text-purple-300 uppercase">{children}</thead>,
										th: ({ children }) => <th className="px-3 py-2 border border-neutral-700 font-semibold">{children}</th>,
										td: ({ children }) => <td className="px-3 py-2 border border-neutral-700 align-top">{children}</td>,
										a: ({ children, href }) => <a href={href} className="text-purple-400 hover:underline">{children}</a>,
										p: ({ children }) => <p className="mb-2 last:mb-0">{children}</p>
										}}
									>
										{m.content}
									</ReactMarkdown>
								)}
							</div>
							{m.trace && (
								<details className="mt-2 text-xs text-neutral-500 w-full max-w-[90%] bg-neutral-900/50 rounded-lg border border-neutral-800">
									<summary className="cursor-pointer hover:text-purple-400 select-none p-2 font-medium focus:outline-none">
										🔍 View AI thinking process & DB queries
									</summary>
									<div className="p-3 border-t border-neutral-800 whitespace-pre-wrap font-mono max-h-[300px] overflow-y-auto custom-scrollbar">
										{m.trace}
									</div>
								</details>
							)}
							{m.records && m.records.length > 0 && isLastAssistantMessage && (
								<details className="mt-2 text-xs text-neutral-500 w-full max-w-[90%] bg-neutral-900/50 rounded-lg border border-neutral-800">
									<summary className="cursor-pointer hover:text-blue-400 select-none p-2 font-medium focus:outline-none">
										📊 View raw data records ({m.records.length} items)
									</summary>
									<div className="p-3 border-t border-neutral-800 whitespace-pre-wrap font-mono max-h-[300px] overflow-y-auto custom-scrollbar">
										{JSON.stringify(m.records, null, 2)}
									</div>
								</details>
							)}
						</div>
					);
				})}
				{loading && (
					<div className="flex flex-col gap-2 w-full max-w-[90%]">
						<div className="flex justify-start items-center gap-3">
							<div className="p-3 rounded-xl bg-neutral-900 border border-neutral-800 text-purple-400 rounded-bl-sm flex items-center gap-2">
								<div className="w-3 h-3 border-2 border-purple-500 border-t-transparent rounded-full animate-spin"></div>
								<span className="animate-pulse text-xs tracking-wide">Investigating...</span>
								<button
									onClick={() => setShowLiveTrace(!showLiveTrace)}
									className="ml-2 text-xs text-neutral-500 hover:text-purple-400 border border-neutral-700 hover:border-purple-500/50 rounded px-2 py-1 transition-colors"
								>
									{showLiveTrace ? "Hide thinking" : "💡 Show thinking"}
								</button>
							</div>
							<button
								onClick={handleStop}
								className="bg-neutral-800 hover:bg-red-500/20 text-neutral-400 hover:text-red-400 border border-neutral-700 hover:border-red-500/50 rounded-full w-8 h-8 flex items-center justify-center transition-colors shrink-0"
								title="Stop generating"
							>
								⏹
							</button>
						</div>
						{showLiveTrace && liveTrace.length > 0 && (
							<div className="bg-neutral-900/80 border border-neutral-800 rounded-lg p-3 font-mono text-xs text-neutral-400 max-h-[200px] overflow-y-auto custom-scrollbar flex flex-col gap-1">
								{liveTrace.map((log, idx) => (
									<div key={idx} className="break-words">{log}</div>
								))}
							</div>
						)}
					</div>
				)}
				<div ref={endOfMessagesRef} />
			</div>

			<div className="p-4 border-t border-neutral-800 bg-neutral-950 flex flex-col gap-3 shrink-0">
				{/* Suggested Action Chips (Hidden for Global if chat has started) */}
				{(!uid && messages.length === 0) || uid ? (
					<div className="flex flex-col gap-2 pb-1">
						<div className="text-xs font-bold text-neutral-500 uppercase tracking-wider">Suggested for you</div>
						{uid ? (
							<>
								<button onClick={() => handleSend("I want to see connected entities")} className="text-xs text-left bg-neutral-900 border border-neutral-800 text-neutral-300 px-3 py-2 rounded-lg hover:bg-neutral-800 hover:text-purple-300 transition-colors shadow-sm">
									Show connected entities
								</button>
								<button onClick={() => handleSend("I want to see the history of this node")} className="text-xs text-left bg-neutral-900 border border-neutral-800 text-neutral-300 px-3 py-2 rounded-lg hover:bg-neutral-800 hover:text-purple-300 transition-colors shadow-sm">
									See history of the node
								</button>
							</>
						) : (
							<>
								<button onClick={() => handleSend("Top 10 financial companies for capital")} className="text-xs text-left bg-neutral-900 border border-neutral-800 text-neutral-300 px-3 py-2 rounded-lg hover:bg-neutral-800 hover:text-purple-300 transition-colors shadow-sm">
									Top 10 financial companies for capital
								</button>
								<button onClick={() => handleSend("Top 10 companies in Dubendorf per capital")} className="text-xs text-left bg-neutral-900 border border-neutral-800 text-neutral-300 px-3 py-2 rounded-lg hover:bg-neutral-800 hover:text-purple-300 transition-colors shadow-sm">
									Top 10 companies in Dubendorf per capital
								</button>
							</>
						)}
					</div>
				) : null}
				<div className="relative flex items-center shrink-0">
					<input
						className="w-full bg-neutral-900 border border-neutral-800 rounded-lg py-3 pl-4 pr-12 text-white placeholder-neutral-500 focus:outline-none focus:border-purple-500 focus:ring-1 focus:ring-purple-500 transition-all text-sm shadow-inner"
						value={input}
						onChange={(e) => setInput(e.target.value)}
						onKeyDown={(e) => e.key === 'Enter' && handleSend()}
						placeholder="Message agent..."
					/>
					<button
						onClick={(e) => handleSend(e)}
						className="absolute right-2 text-purple-500 hover:text-purple-400 hover:bg-purple-500/10 p-1.5 rounded-md transition-all disabled:opacity-30 disabled:hover:bg-transparent"
						disabled={loading || !input.trim()}
					>
						⮑
					</button>
				</div>
			</div>
		</div>
	);
}

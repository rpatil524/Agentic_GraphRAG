"use client";
import React, { useEffect, useRef } from "react";
import dynamic from 'next/dynamic';
import type {
    ForceGraphMethods,
    ForceGraphProps,
    GraphData,
    NodeObject,
} from "react-force-graph-2d";
import { apiUrl } from "../lib/api";
import type { Entity, GraphLink, GraphNode } from "../lib/types";

// Next.js SSR fix for react-force-graph-2d since it relies on browser 'window'
const DynamicForceGraph2D = dynamic(() => import('react-force-graph-2d'), { ssr: false });
const ForceGraph2D = DynamicForceGraph2D as React.ForwardRefExoticComponent<
    ForceGraphProps<GraphNode, GraphLink> &
    React.RefAttributes<ForceGraphMethods<GraphNode, GraphLink>>
>;

interface NetworkGraphProps {
    uid: string;
    setSelectedEntity?: (entity: Entity) => void;
}

interface LoadedGraph {
    uid: string;
    data: GraphData<GraphNode, GraphLink>;
}

const EMPTY_GRAPH: GraphData<GraphNode, GraphLink> = { nodes: [], links: [] };

export default function NetworkGraph({ uid, setSelectedEntity }: NetworkGraphProps) {
    const [loadedGraph, setLoadedGraph] = React.useState<LoadedGraph | null>(null);
    const [selectedNode, setSelectedNode] = React.useState<NodeObject<GraphNode> | null>(null);
    const [graphWidth, setGraphWidth] = React.useState(800);
    const graphRef = useRef<ForceGraphMethods<GraphNode, GraphLink> | null>(null);
    const data = loadedGraph?.uid === uid ? loadedGraph.data : EMPTY_GRAPH;
    const loading = loadedGraph?.uid !== uid;

    useEffect(() => {
        if (!uid) return;
        const controller = new AbortController();
        fetch(apiUrl(`/api/entity/${uid}/graph`))
            .then((res) => {
                if (!res.ok) throw new Error(`Graph request failed with status ${res.status}`);
                return res.json() as Promise<GraphData<GraphNode, GraphLink>>;
            })
            .then((responseData) => {
                if (!controller.signal.aborted) {
                    setLoadedGraph({ uid, data: responseData });
                }
            })
            .catch((error: unknown) => {
                if (!controller.signal.aborted) {
                    console.error("Error fetching graph data:", error);
                    setLoadedGraph({ uid, data: EMPTY_GRAPH });
                }
            });
        return () => controller.abort();
    }, [uid]);

    useEffect(() => {
        const updateWidth = () => {
            const container = document.getElementById('graph-container');
            if (container) {
                setGraphWidth(container.clientWidth);
            }
        };

        updateWidth();
        window.addEventListener('resize', updateWidth);
        const container = document.getElementById('graph-container');
        let observer: ResizeObserver;

        if (container) {
            observer = new ResizeObserver(updateWidth);
            observer.observe(container);
        }

        return () => {
            window.removeEventListener('resize', updateWidth);
            if (observer) observer.disconnect();
        };
    }, []);

    const getNodeColor = (node: NodeObject<GraphNode>) => {
        if (node.id === uid) return "#ef4444"; // Red for target
        switch (node.group) {
            case 1: return "#3b82f6"; // Company - Blue
            case 2: return "#10b981"; // Person - Green
            case 3: return "#f59e0b"; // Event - Yellow
            default: return "#8b5cf6"; // Hub/Other - Purple
        }
    };

    if (loading) return <div className="h-96 w-full flex items-center justify-center text-purple-400 bg-neutral-900 rounded-xl animate-pulse font-medium border border-neutral-800">Visualizing structural relationships...</div>;
    if (data.nodes.length === 0) return <div className="h-96 w-full flex items-center justify-center text-neutral-500 bg-neutral-900 rounded-xl border border-neutral-800">No network data found.</div>;

    return (
        <div className="h-[500px] w-full rounded-xl overflow-hidden border border-neutral-800 bg-neutral-950 relative" id="graph-container">
            <ForceGraph2D
                ref={graphRef}
                graphData={data}
                nodeLabel="name"
                nodeColor={getNodeColor}
                width={graphWidth}
                height={500}
                linkColor={() => "rgba(167, 139, 250, 0.4)"} // Purple-ish links
                linkDirectionalArrowLength={3.5}
                linkDirectionalArrowRelPos={1}
                onNodeClick={(node) => setSelectedNode(node)}
                onEngineStop={() => {
                    if (graphRef.current) {
                        graphRef.current.zoomToFit(400, 20);
                    }
                }}
            />
            <div className="absolute top-4 right-4 bg-neutral-900/80 backdrop-blur text-xs p-3 rounded-lg border border-neutral-800 flex flex-col gap-2">
                <div className="flex items-center gap-2 text-neutral-300"><div className="w-3 h-3 rounded-full bg-red-500"></div> Target</div>
                <div className="flex items-center gap-2 text-neutral-300"><div className="w-3 h-3 rounded-full bg-blue-500"></div> Company</div>
                <div className="flex items-center gap-2 text-neutral-300"><div className="w-3 h-3 rounded-full bg-emerald-500"></div> Person</div>
                <div className="flex items-center gap-2 text-neutral-300"><div className="w-3 h-3 rounded-full bg-amber-500"></div> Event</div>
            </div>

            {selectedNode && (
                <div className="absolute top-4 left-4 bg-neutral-900/90 backdrop-blur border border-neutral-800 p-4 rounded-xl shadow-2xl w-64 z-10 transition-all text-sm">
                    <div className="flex justify-between items-start mb-3">
                        <h4 className="font-bold text-white break-words pr-2">{selectedNode.name || 'Unknown Node'}</h4>
                        <button onClick={() => setSelectedNode(null)} className="text-neutral-500 hover:text-white shrink-0">✕</button>
                    </div>

                    <div className="flex flex-col gap-2">
                        <div className="flex justify-between border-b border-neutral-800 pb-1">
                            <span className="text-neutral-500 font-medium">Type</span>
                            <span className="text-purple-400 font-bold">{selectedNode.label || 'Unknown'}</span>
                        </div>
                        <div className="flex flex-col">
                            <span className="text-neutral-500 font-medium pb-1 mb-1">ID</span>
                            <span className="text-neutral-300 font-mono text-xs break-all bg-neutral-950 p-2 border border-neutral-800 rounded">{selectedNode.id}</span>
                        </div>
                    </div>

                    {setSelectedEntity && selectedNode && selectedNode.group !== 3 && (
                        <button
                            onClick={() => {
                                setSelectedEntity({ uid: selectedNode.id, name: selectedNode.name, real_label: selectedNode.label });
                            }}
                            className="mt-4 w-full bg-purple-600/20 hover:bg-purple-600 text-purple-300 hover:text-white border border-purple-500/30 transition-all font-semibold py-2 rounded-lg flex justify-center items-center gap-2 text-xs"
                        >
                            <span>🔍</span> Open this entity
                        </button>
                    )}
                </div>
            )}
        </div>
    );
}

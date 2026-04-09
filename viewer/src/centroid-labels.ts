import { Vec3 } from 'playcanvas';

import { Element, ElementType } from './element';
import { Events } from './events';
import { Splat } from './splat';

interface CentroidEntry {
    id: string;
    label: string;
    centroid: Vec3;   // raw PLY-space coordinates from object_centroids.json
    div: HTMLDivElement;
}

// Manages HTML div labels overlaid on the 3D canvas, positioned at each object centroid.
class CentroidLabels extends Element {
    private entries: CentroidEntry[] = [];
    private container: HTMLDivElement | null = null;
    private labelsVisible = true;
    private screenVec = new Vec3();
    private worldVec = new Vec3();   // centroid transformed into scene world space
    private pendingJson: File | null = null;  // queued when JSON arrives before PLY finishes

    constructor(private events: Events) {
        super(ElementType.other);
    }

    add() {
        // Overlay container placed inside #canvas-container so it tracks the canvas size.
        this.container = document.createElement('div');
        this.container.id = 'centroid-labels-container';
        Object.assign(this.container.style, {
            position: 'absolute',
            top: '0',
            left: '0',
            width: '100%',
            height: '100%',
            pointerEvents: 'none',
            overflow: 'hidden',
            zIndex: '100'
        });

        document.getElementById('canvas-container')?.appendChild(this.container);

        // Open a native file picker, parse JSON, and load labels.
        this.events.function('centroids.openFilePicker', () => {
            return new Promise<boolean>((resolve) => {
                const input = document.createElement('input');
                input.type = 'file';
                input.accept = '.json';
                input.onchange = async () => {
                    const file = input.files?.[0];
                    if (!file) { resolve(false); return; }
                    try {
                        const text = await file.text();
                        const data = JSON.parse(text) as Record<string, unknown>;
                        this.loadCentroids(data);
                        resolve(true);
                    } catch {
                        resolve(false);
                    }
                };
                input.click();
            });
        });

        // Toggle overall label visibility.
        this.events.on('centroids.setVisible', (visible: boolean) => {
            this.labelsVisible = visible;
            this.entries.forEach(e => {
                e.div.style.visibility = visible ? 'visible' : 'hidden';
            });
            this.scene.forceRender = true;
        });

        // Remove all labels.
        this.events.on('centroids.clear', () => {
            this.clearAll();
            this.scene.forceRender = true;
        });

        // Queue the JSON file delivered alongside the PLY in the same import batch.
        this.events.on('centroids.pendingJson', (file: File) => {
            this.pendingJson = file;
        });

        // Auto-load the queued JSON once the splat element is fully in the scene.
        this.events.on('scene.elementAdded', async (element: Element) => {
            if (element.type !== ElementType.splat || !this.pendingJson) return;
            const file = this.pendingJson;
            this.pendingJson = null;
            try {
                const data = JSON.parse(await file.text()) as Record<string, unknown>;
                this.loadCentroids(data);
            } catch (e) {
                console.warn('[CentroidLabels] Auto-load failed:', e);
            }
        });
    }

    remove() {
        this.clearAll();
        this.container?.remove();
        this.container = null;
    }

    private makeDiv(text: string): HTMLDivElement {
        const div = document.createElement('div');
        div.className = 'centroid-label';

        const tag = document.createElement('span');
        tag.className = 'centroid-label-text';
        tag.textContent = text;

        const dot = document.createElement('div');
        dot.className = 'centroid-label-dot';

        div.appendChild(tag);
        div.appendChild(dot);
        return div;
    }

    loadCentroids(data: Record<string, unknown>) {
        this.clearAll();
        if (!this.container) return;

        for (const [id, rawEntry] of Object.entries(data)) {
            const entry = rawEntry as Record<string, unknown>;
            const label = typeof entry.label === 'string' ? entry.label : `Object ${id}`;

            // Skip background — always ID "0" and/or labelled as background/unlabelled
            if (id === '0' || label.toLowerCase().includes('background')) continue;

            const raw = entry.centroid;
            if (!Array.isArray(raw) || raw.length < 3) continue;

            const div = this.makeDiv(label);
            this.container.appendChild(div);

            this.entries.push({
                id,
                label,
                centroid: new Vec3(raw[0] as number, raw[1] as number, raw[2] as number),
                div
            });
        }

        this.scene.forceRender = true;
        this.events.fire('centroids.loaded', { count: this.entries.length });
    }

    clearAll() {
        this.entries.forEach(e => e.div.remove());
        this.entries = [];
    }

    // Called every frame (when rendering is active) by scene.onPreRender().
    onPreRender() {
        if (!this.scene || !this.entries.length || !this.container || !this.labelsVisible) return;

        const splats = this.scene.getElementsByType(ElementType.splat) as Splat[];
        if (splats.length === 0) return;
        const splatTransform = splats[0].entity.getWorldTransform();

        const camera = this.scene.camera;
        const canvas = this.scene.canvas;
        const w = canvas.clientWidth;
        const h = canvas.clientHeight;
        const screen = this.screenVec;
        const worldPos = this.worldVec;
        const camForward = camera.forward;
        const camPos = camera.position;

        for (const entry of this.entries) {
            splatTransform.transformPoint(entry.centroid, worldPos);

            const dx = worldPos.x - camPos.x;
            const dy = worldPos.y - camPos.y;
            const dz = worldPos.z - camPos.z;
            const dot = dx * camForward.x + dy * camForward.y + dz * camForward.z;
            if (dot < 0) {
                entry.div.style.display = 'none';
                continue;
            }

            camera.worldToScreen(worldPos, screen);

            entry.div.style.display = 'block';
            entry.div.style.left = `${screen.x * w}px`;
            entry.div.style.top = `${screen.y * h}px`;
        }
    }

    get count() {
        return this.entries.length;
    }
}

export { CentroidLabels };
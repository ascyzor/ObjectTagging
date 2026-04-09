import { BooleanInput, Button, Container, Label } from '@playcanvas/pcui';

import { Events } from '../events';
import { Tooltips } from './tooltips';

class CentroidPanel extends Container {
    constructor(events: Events, tooltips: Tooltips, args = {}) {
        args = {
            ...args,
            id: 'centroid-panel',
            class: 'panel'
        };

        super(args);

        // Prevent pointer events from passing through to the 3D scene.
        ['pointerdown', 'pointerup', 'pointermove', 'wheel', 'dblclick'].forEach((name) => {
            this.dom.addEventListener(name, (e: Event) => e.stopPropagation());
        });

        // --- Header ---
        const header = new Container({ class: 'panel-header' });
        const headerIcon = new Label({ text: '\uE344', class: 'panel-header-icon' });
        const headerLabel = new Label({ text: 'Object Labels', class: 'panel-header-label' });
        header.append(headerIcon);
        header.append(headerLabel);
        this.append(header);

        // --- Load button ---
        const loadRow = new Container({ class: 'centroid-row' });
        const loadButton = new Button({ text: 'Load Centroids JSON', class: 'centroid-load-btn' });
        loadRow.append(loadButton);
        this.append(loadRow);

        // --- Visibility toggle ---
        const visRow = new Container({ class: 'centroid-row' });
        const visLabel = new Label({ text: 'Show Labels', class: 'centroid-row-label' });
        const visToggle = new BooleanInput({ value: true, class: 'centroid-row-toggle' });
        visRow.append(visLabel);
        visRow.append(visToggle);
        this.append(visRow);

        // --- Status ---
        const status = new Label({ text: 'No centroids loaded', class: 'centroid-status' });
        this.append(status);

        // Button → open file picker
        loadButton.on('click', async () => {
            const ok = await events.invoke('centroids.openFilePicker');
            if (!ok) {
                status.text = 'Error: could not load JSON';
            }
        });

        // Show pending state when JSON is queued but PLY hasn't finished yet
        events.on('centroids.pendingJson', () => {
            status.text = 'Auto-loading labels after PLY loads...';
        });

        // Update status when labels are loaded (manual or auto)
        events.on('centroids.loaded', ({ count }: { count: number }) => {
            status.text = `${count} object label${count !== 1 ? 's' : ''} loaded`;
        });

        // Visibility toggle → broadcast event
        visToggle.on('change', (value: boolean) => {
            events.fire('centroids.setVisible', value);
        });

        tooltips.register(loadButton, 'Load object_centroids.json from your pipeline output');
        tooltips.register(visToggle, 'Toggle label visibility');
    }
}

export { CentroidPanel };
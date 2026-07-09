import { defineStore } from "pinia";

export interface Toast {
  id: number;
  message: string;
  kind: "info" | "error";
}

let nextId = 1;

/** Toasts rise from the Bench rail; 4 s, max 3 on screen (design spec §6). */
export const useToastStore = defineStore("toasts", {
  state: () => ({ items: [] as Toast[] }),
  actions: {
    push(message: string, kind: Toast["kind"] = "info") {
      const toast = { id: nextId++, message, kind };
      this.items.push(toast);
      if (this.items.length > 3) this.items.shift();
      setTimeout(() => this.dismiss(toast.id), 4000);
    },
    dismiss(id: number) {
      this.items = this.items.filter((t) => t.id !== id);
    },
  },
});

/** Raw base64 (no data-URI prefix) of a picked or dropped file. */
export function readFileBase64(file: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () =>
      reject(reader.error ?? new Error("Could not read the file."));
    reader.onload = () => {
      const result = String(reader.result ?? "");
      const comma = result.indexOf(",");
      if (comma < 0) reject(new Error("The file could not be encoded."));
      else resolve(result.slice(comma + 1));
    };
    reader.readAsDataURL(file);
  });
}

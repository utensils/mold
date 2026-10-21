import { apiJsonTo } from "./client";
import { assertVideoJobOwner, type VideoJobOwner } from "./videoJobs";
/** An exact active Library listing proves existence, never an offline snapshot. */
export async function videoLibraryOutputExists(
  owner: VideoJobOwner,
  filename: string,
  isCurrent: () => boolean = () => true,
): Promise<boolean> {
  const check = () => {
    if (!isCurrent())
      throw new Error("This video's machine changed. Check its Library again.");
  };
  check();
  await assertVideoJobOwner(owner);
  check();
  const rows = await apiJsonTo<Array<{ filename: string }>>(
    owner.target,
    `/api/gallery?filename=${encodeURIComponent(filename)}`,
    { headers: { "x-mold-destination-instance": owner.instanceId! } },
  );
  check();
  await assertVideoJobOwner(owner);
  check();
  if (
    !Array.isArray(rows) ||
    rows.length > 1 ||
    rows.some((row) => !row || row.filename !== filename)
  )
    throw new Error(
      "This machine returned an invalid Library result. Refresh before opening the video.",
    );
  return rows.length === 1;
}

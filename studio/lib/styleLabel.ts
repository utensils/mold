import { modelDisplayName, type DisplayableModel } from "./modelDisplay";
import { familyLabel } from "./modelFamily";

/** One friendly style name across web and native pickers; IDs remain wire values. */
export function styleDisplayName(
  model: DisplayableModel & { family: string },
): string {
  const display = modelDisplayName(model);
  const description = model.description?.trim();
  return (
    (description && description !== model.name ? description : null) ??
    (display !== model.name ? display : familyLabel(model.family))
  );
}

/** Native option text retains the exact runnable ID beside the friendly name. */
export function styleLabel(
  model: DisplayableModel & { family: string },
): string {
  return [styleDisplayName(model), model.name]
    .filter((value, index, values) => value && values.indexOf(value) === index)
    .join(" · ");
}

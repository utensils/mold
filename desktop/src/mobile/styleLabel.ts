import { modelDisplayName, type DisplayableModel } from "@studio/lib/modelDisplay";
import { familyLabel } from "@studio/lib/modelFamily";

/** Native options keep the runnable ID beside a human description. */
export function mobileStyleLabel(model: DisplayableModel & { family: string }): string {
  const display = modelDisplayName(model);
  const description = model.description?.trim();
  const friendly =
    (description && description !== model.name ? description : null) ??
    (display !== model.name ? display : familyLabel(model.family));
  return [friendly, model.name]
    .filter((value, index, values) => value && values.indexOf(value) === index)
    .join(" · ");
}

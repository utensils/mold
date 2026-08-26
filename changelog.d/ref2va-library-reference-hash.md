- **MiniMax H3 Ref2VA Library references now queue correctly.** Mold binds each
  selected image to its content hash before placement, so Library and local-file
  references work across web, desktop, iPhone, and Android instead of being
  refused with a missing SHA-256 error; hosts without an API key use Mold's
  validated inline-reference path instead of being forced into authenticated
  uploads.

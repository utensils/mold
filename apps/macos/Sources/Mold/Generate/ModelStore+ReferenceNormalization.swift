import MoldClient

extension ModelStore {
    /// Filter the existing adopted/default host order before creating any scoped media.
    func referenceNormalizationHost(model: String?, hardSelection: MoldHost.ID?, adopted: MoldHost.ID?) -> MoldHost? {
        guard let model else { return nil }
        let eligible = { (host: MoldHost) in
            self.hosts.isUp(host) && self.model(named: model, on: host.id) != nil
                && VideoNormalizationPolicy.capability(self.hosts.capabilities(of: host)?.referenceUploads, apiKey: host.apiKey) != nil
        }
        if let hardSelection { return hosts.host(hardSelection).flatMap { eligible($0) ? $0 : nil } }
        let ordered = [adopted.flatMap(hosts.host), hosts.preferredHost].compactMap { $0 } + hosts.hosts
        return ordered.first(where: eligible)
    }
}

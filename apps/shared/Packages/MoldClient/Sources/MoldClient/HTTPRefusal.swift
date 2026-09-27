import Foundation

/// What a non-2xx answer means, given whatever body came with it.
///
/// The body is mold's `APIError` envelope where there is one, and the
/// machine's own plain sentence where there is not (`RefusalBody`). Both are
/// better than "the machine answered with an error (400)", which is what a
/// status code alone can say.
enum HTTPRefusal {
    static func check(_ http: HTTPURLResponse, _ data: Data) throws {
        guard (200..<300).contains(http.statusCode) else {
            if http.statusCode == 401 { throw MoldClientError.unauthorized }
            let api = try? MoldJSON.decoder.decode(APIError.self, from: data)
            if let refusal = api?.license,
               api?.code == LicenseCode.notAccepted || api?.code == LicenseCode.termsMismatch {
                throw MoldClientError.licenseRequired(refusal, mismatch: api?.code == LicenseCode.termsMismatch)
            }
            throw MoldClientError.http(
                status: http.statusCode,
                code: api?.code,
                message: api?.error ?? RefusalBody.plainMessage(data)
            )
        }
    }
}
